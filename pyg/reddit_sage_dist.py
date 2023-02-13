"""
Edit by HappySky12

Official : https://github.com/pyg-team/pytorch_geometric/blob/master/examples/multi_gpu/distributed_sampling.py (PyG example)

This code is for (distributed) data parallel training for multiple GPUs.

Since we are not using such graph partitioning scheme, only using neighbor sampling, this script will run in only mini-batch manner. 
"""

import argparse
import copy
import os 

import numpy as np
import pandas as pd

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel

import torch_geometric.transforms as T

from torch_geometric.datasets import Reddit
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader

parser = argparse.ArgumentParser(description='Reddit (GraphSAGE_Distributed)')

parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--hidden_channels', type=int, default=128) 
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--fanout', type=int, nargs='+', help="# of fanouts. Should be len(fanout) == len(num_layers).", required=True)
parser.add_argument('--batch_size', type=int, help="# of anchor node in each batch. # of batch will be 'len(num_nodes)/len(batch_size)'", required=True)

args = parser.parse_args()

if args.fanout is None or args.batch_size is None:
    raise Exception ("Should specify '--fanout' and '--batch_size'")

if len(args.fanout) != args.num_layers:
    raise Exception (f"Fanout length should be same with 'num_layers' (len(fanout)({len(args.fanout)}) != num_layers({args.num_layers})).")

print(args)

class SAGE_Dist(torch.nn.Module):
    """Mini-batch GraphSAGE"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(SAGE_Dist, self).__init__()

        self.conv_layers = torch.nn.ModuleList()
        self.conv_layers.append(SAGEConv(in_channels, hidden_channels, aggr='mean'))            # 1st layer
        for _ in range(num_layers - 2):
            self.conv_layers.append(SAGEConv(hidden_channels, hidden_channels, aggr='mean'))    # hidden layers
        self.conv_layers.append(SAGEConv(hidden_channels, out_channels, aggr='mean'))           # last layer

        self.dropout = dropout
        self.num_layers = num_layers

    def reset_parameters(self):
        for conv in self.conv_layers:
            conv.reset_parameters()

    # Since we are using SparseTensor, not 'edge_index', we will use adj_t for message passing.
    # SparseTensor accelerates message passing. (https://github.com/pyg-team/pytorch_geometric/discussions/4901)
    def forward(self, x, adj_t):
        for conv in self.conv_layers[:-1]: # message passing from 1st layer to hidden layers
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv_layers[-1](x, adj_t) # message passing from last hidden layer to last layer
        return x
    
    @torch.no_grad()
    def inference(self, x_all, device, subgraph_loader):
        # progress_bar = tqdm(total=len(subgraph_loader.dataset) * len(self.conv_layers))
        # progress_bar.set_description('Evaluating...')

        # At each layer, GraphSAGE takes **all 1-hop neighbors** to compute node representations.
        # This leads to faster computation in contrast to immediately computing final representations of each batch.
        for i, conv in enumerate(self.conv_layers):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(device)   # access to original index, since index in batch is differ from original.
                x = conv(x, batch.adj_t.to(device))                 # message_passing for all nodes (in mini-batch).
                if i < len(self.conv_layers) - 1:
                    x = F.relu(x)
                xs.append(x[:batch.batch_size].cpu())               # will move it to main memory for inference.
                # progress_bar.update(batch.batch_size)
            x_all = torch.cat(xs, dim=0)
        # progress_bar.close()
        return x_all

def run(rank, world_size, dataset):
    """Run GraphSAGE in Distributed Data Parallel (DDP) Manner."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # world_size is same as # of GPUs (in single machine - multi GPU).
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    data = dataset[0]

    # Send node features and labels to device for faster access during sampling.
    data = data.to(rank, 'x', 'y')

    # Split training indices into chunks as 'world_size'. 
    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    train_idx = train_idx.split(train_idx.size(0) // world_size)[rank]

    train_loader = NeighborLoader(
        data = data,
        input_nodes = train_idx,            # will make mini-batches in each GPU, with partitioned train nodes.
        num_neighbors = args.fanout,
        shuffle = True,
        batch_size = args.batch_size,       # nodes in data[train_idx] is anchor nodes to make computation graph in each mini-batch, and # of anchor node in each mini-batch is same as 'batch_size'.
        num_workers = 6,
        persistent_workers = True
    )

    # Create 1-hop evaluation neighbor loader.
    # This loader will used in inference.
    # At each layer, GraphSAGE takes **all 1-hop neighbors** to compute node representations.
    # This leads to faster computation in contrast to immediately computing final representations of each batch.
    # Refer : Inductive Representation Learning on Large Graphs [Hamilton et al., 2017] (GraphSAGE)
    if rank == 0:
        """Define neighbor loader for inference in 1st GPU"""
        subgraph_loader = NeighborLoader(
            data = copy.copy(data),
            input_nodes = None,             # will make mini-batches with all nodes.
            num_neighbors = [-1],           # will consider all 1-hop neighbors to compute node representation.
            shuffle = False,
            batch_size = args.batch_size,
            num_workers = 6,
            persistent_workers = True
        )

        # We don't need to maintin these features during evaluation, so delete it.
        del subgraph_loader.data.x, subgraph_loader.data.y

        # Add global node index information for mini-batch inference
        subgraph_loader.data.num_nodes = data.num_nodes
        subgraph_loader.data.n_id = torch.arange(data.num_nodes)
    
    # Move model to GPU
    model = SAGE_Dist(
        in_channels = dataset.num_features,
        hidden_channels = args.hidden_channels,
        out_channels = dataset.num_classes,
        num_layers = args.num_layers,
        dropout = args.dropout
    ).to(rank)

    # and wrap it with DDP Module.
    model = DistributedDataParallel(model, device_ids=[rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # FIXME: Official Repo is not using train/test function, maybe we can functionize it?
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()

            y = batch.y[:batch.batch_size]
            y_hat = model(batch.x, batch.adj_t.to(rank))[:batch.batch_size]

            loss = F.cross_entropy(y_hat, y)

            loss.backward()
            optimizer.step()
        
        # Must synchronize all GPUs.
        dist.barrier()

        # We evaluate on a single GPU for now.
        if rank == 0:
            print(f'Epoch {epoch:03d}, Loss: {loss:.4f}')

            model.eval()
            with torch.no_grad():
                out = model.module.inference(data.x, rank, subgraph_loader)
            result = out.argmax(dim=-1) == data.y.to(out.device)

            train_acc = int(result[data.train_mask].sum()) / int(data.train_mask.sum())
            val_acc = int(result[data.val_mask].sum()) / int(data.val_mask.sum())
            test_acc = int(result[data.test_mask].sum()) / int(data.test_mask.sum())

            print(f'Epoch {epoch:03d}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
        
        # Must synchronize all GPUs.
        dist.barrier()
    
    dist.destroy_process_group()

if __name__ == '__main__':
    dataset = Reddit(root='../dataset/Reddit/', transform=T.ToSparseTensor())

    world_size = torch.cuda.device_count()

    print(f'Using {world_size} GPUs...')

    mp.spawn(run, args=(world_size, dataset), nprocs=world_size, join=True) # join : Perform a blocking join on all processes.