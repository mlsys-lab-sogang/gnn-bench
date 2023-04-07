r"""
Data-parallel training script for GraphSAGE with Reddit dataset.
The original source from DGL is available at:
    https://github.com/dmlc/dgl/tree/master/examples/pytorch/graphsage/dist
"""
import argparse
import socket
import os
import logging

from contextlib import contextmanager
from datetime import datetime
from tqdm.auto import tqdm

import pandas as pd
import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

import dgl
from dgl.nn.pytorch import SAGEConv
from dgl.dataloading import NeighborSampler, DistNodeDataLoader

def parse_args():
    parser = argparse.ArgumentParser(description="Data-parallel training of GraphSAGE with Reddit")

    # args for distributed setting.
    parser.add_argument('--graph_name', type=str, help='Name of partitioned graph. This name is made from `partition_graph.py`\'s output.')
    parser.add_argument('--part_id', type=int, help="Graph partition id to use.")
    parser.add_argument('--ip_config', type=str, help="File(.txt) for IP configuration. File should have **all** participating cluster's IP (& Port) address.")
    parser.add_argument('--part_config', type=str, help="Path of partition config file(.json).")
    parser.add_argument('--standalone', action='store_true', help="Run in standalone mode. Usually used for testing.")
    parser.add_argument('--nnodes', type=int, default=2, help="Total number of machines participating in distributed training.")
    parser.add_argument('--nprocs', type=int, default=4, help="Total number of GPUs in current machine.")
    parser.add_argument('--node_id', type=int, required=True, help="Current machine's ID for distributed setting. (0 ~ 'num_cluster - 1')")

    # args for train setting.
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--fanout', type=int, nargs='+', required=True)

    args = parser.parse_args()

    assert len(args.fanout) == args.num_layers, \
        f"Length of fanout should be same with number of layers: len(fanout)({len(args.fanout)}) != num_layers({args.num_layers}))."

    return args

def load_subtensor(data, seeds, input_nodes, device, load_feature=True):
    r"""Copy features and labels of a set of nodes, send it to GPU"""
    batch_inputs = (data.ndata['features'][input_nodes].to(device) if load_feature else None)
    batch_labels = data.ndata['labels'][seeds].to(device)

    return batch_inputs, batch_labels

class DistSAGE(torch.nn.Module):
    r"Mini-batch GraphSAGE"
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(DistSAGE, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_size = hidden_channels
        self.output_size = out_channels

        self.conv_layers = torch.nn.ModuleList()
        
        self.conv_layers.append(SAGEConv(in_channels, hidden_channels, aggregator_type='mean'))
        for _ in range(self.num_layers - 2):
            self.conv_layers.append(SAGEConv(hidden_channels, hidden_channels, aggregator_type='mean'))
        self.conv_layers.append(SAGEConv(hidden_channels, out_channels, aggregator_type='mean'))
    
    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.conv_layers, blocks)):
            h = layer(block, h)                 # message passing between layers
            if l != len(self.conv_layers) - 1:  # apply non-linear & dropout in hidden layers
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h
    
    def inference(self, data, device, batch_size):
        r"""
        Distributed layer-wise inference.

        Take **all 1-hop neighbors** to compute nore representations for each layer.
        (layer-wise inference to get all node embeddings.)
        """
        features = data.ndata['features']

        # split input nodes based on 'partition book', and return a subset of nodes for local rank.
        # more details : https://docs.dgl.ai/generated/dgl.distributed.node_split.html#dgl.distributed.node_split 
        node_ids = dgl.distributed.node_split(np.arange(data.num_nodes()), data.get_partition_book(), force_even=True)

        # access to distributed tensor, shareded and stored in machines.
        # more details : https://docs.dgl.ai/api/python/dgl.distributed.html#distributed-tensor
        y = dgl.distributed.DistTensor(shape=(data.num_nodes(), self.hidden_size), dtype=torch.float32, name='h', persistent=True)

        for i, layer in enumerate(self.conv_layers):
            if i == len(self.conv_layers) -1:
                y = dgl.distributed.DistTensor(shape=(data.num_nodes(), self.output_size), dtype=torch.float32, name='h_last', persistent=True)
            
            print(f'|V| = {data.num_nodes()}')

            sampler = NeighborSampler(fanouts=[-1])
            dataloader = DistNodeDataLoader(
                g = data,
                nids = node_ids,
                graph_sampler = sampler,
                batch_size = batch_size,
                shuffle = False,
                drop_last = False
            )

            for input_nodes, output_nodes, blocks in tqdm(dataloader):
                block = blocks[0].to(device)

                # fetch nodes for computation from blocks.
                h = features[input_nodes].to(device)
                h_dst = h[: block.number_of_dst_nodes()]

                # calculate node representation layer by layer.
                h = layer(block, (h, h_dst))
                if i != len(self.conv_layers) -1:
                    h = F.relu(h)
                    h = F.dropout(h, p=self.dropout)

                y[output_nodes] = h.cpu()
            
            features = y
            data.barrier()
        
        return y
    
    @contextmanager
    def join(self):
        r"""dummy join for standalone mode"""
        yield
    
def compute_acc(pred, labels):
    r"""Compute accuracy of prediction given the labels."""
    labels = labels.long()

    return (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)

@torch.no_grad()
def evaluate(model, data, labels, val_id, test_id, batch_size, device):
    model.eval()

    pred = model.inference(data, device, batch_size)

    return compute_acc(pred[val_id], labels[val_id]), compute_acc(pred[test_id], labels[test_id])

def run(args, device, data):
    r"""Train GraphSAGE in distributed manner."""

    # unpack data
    train_id, val_id, test_id, n_classes, data = data

    # define sampler and loader
    sampler = NeighborSampler(fanouts=args.fanout)
    dataloader = DistNodeDataLoader(
        g = data,
        nids = train_id,
        graph_sampler = sampler,
        batch_size = args.batch_size,
        shuffle = True,
        drop_last = False
    )

    # define model and optimizer
    model = DistSAGE(
        in_channels = data.ndata['features'].shape[1],
        hidden_channels = args.hidden_channels,
        out_channels = n_classes,
        num_layers = args.num_layers,
        dropout = args.dropout
    ).to(device)

    # check if script is running on standalone setting or distributed setting.
    if not args.standalone:
        model = DistributedDataParallel(model, device_ids=[device], output_device=device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # training loop
    for epoch in range(args.epochs):
        num_seeds, num_inputs = 0, 0
        total_train_acc = 0
        total_train_loss = 0

        # batch loop
        # for DDP.join, see : https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.join 
        with model.join():
            for step, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
                optimizer.zero_grad()

                # fetch features and labels.
                # batch_inputs, batch_labels = load_subtensor(data=data, seeds=output_nodes, input_nodes=input_nodes, device='cpu')
                batch_inputs, batch_labels = load_subtensor(data=data, seeds=output_nodes, input_nodes=input_nodes, device=device)

                batch_labels = batch_labels.long()

                num_seeds += len(blocks[-1].dstdata[dgl.NID])
                num_inputs += len(blocks[0].srcdata[dgl.NID])

                # move data to device
                blocks = [block.to(device) for block in blocks]
                batch_inputs = batch_inputs.to(device)
                batch_labels = batch_labels.to(device)

                # compute loss and prediction
                batch_pred = model(blocks, batch_inputs)
                loss = F.cross_entropy(batch_pred, batch_labels)

                loss.backward()
                optimizer.step()

                train_acc = compute_acc(batch_pred, batch_labels)
                total_train_acc += train_acc
                total_train_loss += loss.item()
        
        train_acc = total_train_acc / (step + 1)
        total_train_loss = total_train_loss / (step + 1)
        val_acc, test_acc = evaluate(
            model = model if args.standalone else model.module,
            data = data,
            labels = data.ndata['labels'],
            val_id = val_id,
            test_id = test_id,
            batch_size = args.batch_size,
            device = device
        )

        print(f'Epoch: {epoch:03d}, ' 
                f'GPU(Partition): {device}, '
                f'Loss: {total_train_loss:.4f}, '
                f'Train Acc: {train_acc:.4f}, '
                f'Val Acc: {val_acc:.4f}, '
                f'Test Acc: {test_acc:.4f}')

def main(args):
    print(socket.gethostname(), "Initializing DistDGL")

    dgl.distributed.initialize(ip_config=args.ip_config, net_type='socket')

    if not args.standalone:
        print(socket.gethostname(), "Initializing DistDGL process group")
        dist.init_process_group(backend='nccl')
    
    print(socket.gethostname(), "Initializing DistGraph")
    data = dgl.distributed.DistGraph(graph_name=args.graph_name, part_config=args.part_config)

    print(socket.gethostname(), f'rank: {data.rank()}')

    # partition book contains informations of partitioned graph.
    part_book = data.get_partition_book()
    if 'trainer_id' in data.ndata:
        train_id = dgl.distributed.node_split(
            nodes = data.ndata['train_mask'],
            partition_book = part_book,
            force_even = True,
            node_trainer_ids = data.ndata['trainer_id']
        )
        val_id = dgl.distributed.node_split(
            nodes = data.ndata['val_mask'],
            partition_book = part_book,
            force_even = True,
            node_trainer_ids = data.ndata['trainer_id']
        )
        test_id = dgl.distributed.node_split(
            nodes = data.ndata['test_mask'],
            partition_book = part_book,
            force_even = True,
            node_trainer_ids = data.ndata['trainer_id']
        )
    else:
        train_id = dgl.distributed.node_split(
            nodes = data.ndata['train_mask'],
            partition_book = part_book,
            force_even = True
        )
        val_id = dgl.distributed.node_split(
            nodes = data.ndata['val_mask'],
            partition_book = part_book,
            force_even = True
        )
        test_id = dgl.distributed.node_split(
            nodes = data.ndata['test_mask'],
            partition_book = part_book,
            force_even = True
        )
    
    # To check each partitioned graph's number of nodes, get entire graph's node id.
    local_nid = part_book.partid2nids(part_book.partid).detach().numpy()
    print(f'Part: {data.rank()}, '
          f'Train: {len(train_id)} '
          f'(local: {np.intersect1d(train_id.numpy(), local_nid)}), '
          f'Val: {len(val_id)} '
          f'(local: {np.intersect1d(val_id.numpy(), local_nid)}), '
          f'Test: {len(test_id)} '
          f'(local: {np.intersect1d(test_id.numpy(), local_nid)})')

    del local_nid

    # TODO: Does it correct? Need to check.
    # get device id (global rank in PyTorch DDP)
    device_id = data.rank() % args.nprocs
    device = torch.device(f'cuda:{device_id}')

    labels = data.ndata['labels'][np.arange(data.num_nodes())]
    n_classes = len(torch.unique(labels[torch.logical_not(torch.isnan(labels))]))
    del labels

    print(f'Number of Labels: {n_classes}')

    # Pack data
    data = train_id, val_id, test_id, n_classes, data

    run(args, device, data)

    print('Parent ends')

if __name__ == "__main__":
    args = parse_args()

    args.world_size = args.nnodes * args.nprocs

    print(f'Args: {args}')
    print(f'Using {args.world_size} GPUs')

    main(args)