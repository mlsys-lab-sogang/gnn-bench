"""
Edit by HappySky12

Official : https://github.com/pyg-team/pytorch_geometric/blob/master/examples/multi_gpu/distributed_sampling.py (PyG example on Reddit dataset)

This code is for (distributed) data parallel training for multiple GPUs.

Since we are not using such graph partitioning scheme, only using neighbor sampling, this script will run in only mini-batch manner. 

(0215) FIX : This script can't run 'adj_t' (SparseTensor) as input. Maybe it's some PyTorch's issue, so fixed 'adj_t' to 'edge_index' as official code.
"""
import argparse
import copy
import os 

# Since OGB stucks in last import, moved it to here.
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

import numpy as np
import pandas as pd

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel

from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader

def parse_args():
    parser = argparse.ArgumentParser(description='OGBN-Products (GraphSAGE_Distributed)')

    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=128)     # 256 occurs CUDA OOM
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--fanout', type=int, nargs='+', help="# of fanouts. Should be len(fanout) == len(num_layers).", required=True)
    parser.add_argument('--batch_size', type=int, help="# of anchor node in each batch. # of batch will be 'len(num_nodes)/len(batch_size)'", required=True)

    args = parser.parse_args()

    if args.fanout is None or args.batch_size is None:
        raise Exception ("Should specify '--fanout' and '--batch_size'")

    if len(args.fanout) != args.num_layers:
        raise Exception (f"Fanout length should be same with 'num_layers' (len(fanout)({len(args.fanout)}) != num_layers({args.num_layers})).")

    print(args)

    return args

class SAGE_Dist(torch.nn.Module):
    """Mini-batch GraphSAGE"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(SAGE_Dist, self).__init__()

        self.conv_layers = torch.nn.ModuleList()
        self.conv_layers.append(SAGEConv(in_channels, hidden_channels, aggr='mean'))            # 1st layer
        for _ in range(num_layers -2):
            self.conv_layers.append(SAGEConv(hidden_channels, hidden_channels, aggr='mean'))    # hidden layers
        self.conv_layers.append(SAGEConv(hidden_channels, out_channels, aggr='mean'))           # last layer

        self.dropout = dropout
        self.num_layers = num_layers

    def reset_parameters(self):
        for conv in self.conv_layers:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for conv in self.conv_layers[:-1]:      # message passing from 1st layer to hidden layers
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv_layers[-1](x, edge_index) # message passing from last hidden layer to last layer
        return x

    @torch.no_grad()
    def inference(self, x_all, device, subgraph_loader):
        # At each layer, GraphSAGE takes **all 1-hop neighbors** to compute node representations.
        # This leads to faster computation in contrast to immediately computing final representations of each batch.
        for i, conv in enumerate(self.conv_layers):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(device)       # access to original(global) index, since index in batch is differ from original.
                x = conv(x, batch.edge_index.to(device))                # message passing for all 1-hop nodes in mini-batch
                if i < len(self.conv_layers) - 1:
                    x = F.relu(x)
                xs.append(x[:batch.batch_size].cpu())                   # will move it to main memory for inference.
            x_all = torch.cat(xs, dim=0)
        return x_all

# pre-defined Evaluator
evaluator = Evaluator(name='ogbn-products')

# This function will copied to each GPU device.
# Process will made as # of GPUs, and each process will take each GPU.
# And each process will execute this function by using each GPU, in parallel manner.
def run(rank, world_size, dataset, args):
    """Run GraphSAGE in Distributed Data Parallel (DDP) Manner."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # world_size is same as # of GPUs (in single machine with multi GPU)
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    # Data(num_nodes=2449029, x=[2449029, 100], y=[2449029, 1], adj_t=[2449029, 2449029, nnz=123718280])
    data = dataset[0]

    # OGB dataset has no mask. Instead they provides 'split_index()'
    # So we can split indexes and use them as mask index.
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train']

    # Since dim of 'data.y' is (2449029, 1), transform it to (2449029,)
    data.y = data.y.squeeze()

    # Send node features and labels to device for faster access during training.
    data = data.to(rank, 'x', 'y')

    # Split training indices into chunks as 'world_size'
    # Unlike 'reddit_sage_dist.py', we can access to idx by using 'split_idx['~']'. (data.train_mask is train_idx in here.)
    train_idx_each_gpu = train_idx.nonzero(as_tuple=False).view(-1)
    train_idx_each_gpu = train_idx_each_gpu.split(train_idx_each_gpu.size(0) // world_size)[rank]

    train_loader = NeighborLoader(
        data = data,
        input_nodes = train_idx_each_gpu,       # will make mini-batches in each GPU, with partitioned train nodes.
        num_neighbors = args.fanout,
        shuffle = True,
        drop_last = True,
        batch_size = args.batch_size,           # nodes in data[train_idx] is anchor nodes to make computation graph in each mini-batch, and # of anchor nodes in each mini-batch is same as 'batch_size'.
        num_workers = 12,
        persistent_workers = True
    )

    # Create 1-hop evaluation neighbor loader.
    # This loader will used in inference.
    # At each layer, GraphSAGE takes **all 1-hop neighbors** to compute node representations.
    # This leads to faster computation in contrast to immediately computing final representations of each iteration.
    # Refer : Inductive Representation Learning on Large Graphs [Hamilton et al., 2017] (GraphSAGE)
    if rank == 0:
        """Define neighbor loader for inference in 1st (main) process."""
        subgraph_loader = NeighborLoader(
            data = copy.copy(data),
            input_nodes = None,             # will make mini-batches with all nodes.
            num_neighbors = [-1],           # will consider all 1-hop neighbors to compute node representations.
            shuffle = False,
            batch_size = args.batch_size,
            num_workers = 12,
            persistent_workers = True
        )

        # We don't need to maintain these features during evalutation, so delete it.
        del subgraph_loader.data.x, subgraph_loader.data.y

        # Add global node index information for mini-batch inference.
        subgraph_loader.data.num_nodes = data.num_nodes
        subgraph_loader.data.n_id = torch.arange(data.num_nodes)
    
    # Move model to each GPU
    model = SAGE_Dist(
        in_channels = dataset.num_features,
        hidden_channels = args.hidden_channels,
        out_channels = dataset.num_classes,
        num_layers = args.num_layers,
        dropout = args.dropout
    ).to(rank)

    # and wrap it with DDP module.
    model = DistributedDataParallel(model, device_ids=[rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if rank == 0:
        print('Enter training...') # just for checking.

    for run in range(args.runs):
        model.module.reset_parameters()

        for epoch in range(args.epochs):
            model.train()
            
            total_loss = total_examples = 0
            for batch in train_loader:
                optimizer.zero_grad()

                y = batch.y[:batch.batch_size]
                y_hat = model(batch.x, batch.edge_index.to(rank))[:batch.batch_size]

                # loss = F.nll_loss(y_hat, y)         # Each process has its own loss, we need to aggregate it for calculating overall loss.
                loss = F.cross_entropy(y_hat, y)      # Why does nll_loss returns - values?

                loss.backward()
                optimizer.step()

                total_loss += float(loss) * batch.batch_size
                total_examples += batch.batch_size
            
            loss_after_batch = total_loss / total_examples
            
            # print(f'Epoch: {epoch:03d}, GPU: {rank}, Loss: {loss:.4f}')
            print(f'Epoch: {epoch:03d}, GPU: {rank}, Loss: {loss_after_batch:.4f}')
            
            # Must synchronize all GPUs.
            dist.barrier()

            # We evaluate on a single process now.
            # We can aggregate each GPU's loss, and calcuate aggregated loss.
            if rank == 0:
                model.eval()
                with torch.no_grad():
                    out = model.module.inference(data.x, rank, subgraph_loader)

                    y_true = data.y.cpu().unsqueeze(-1)
                    y_pred = out.argmax(dim=-1, keepdim=True)

                    train_acc = evaluator.eval({
                        'y_true': y_true[split_idx['train']],
                        'y_pred': y_pred[split_idx['train']],
                    })['acc']
                    val_acc = evaluator.eval({
                        'y_true': y_true[split_idx['valid']],
                        'y_pred': y_pred[split_idx['valid']],
                    })['acc']
                    test_acc = evaluator.eval({
                        'y_true': y_true[split_idx['test']],
                        'y_pred': y_pred[split_idx['test']],
                    })['acc']
                
                if epoch % args.log_steps == 0:
                    print(f'Run: {run + 1:02d}, '
                        f'Epoch: {epoch:03d}, '
                        # f'Loss: {loss:.4f}, '
                        f'Loss: {loss_after_batch:.4f}, '
                        f'Train: {100 * train_acc:.4f}%, '
                        f'Valid: {100 * val_acc:.4f}% '
                        f'Test: {100 * test_acc:.4f}%,')
            
            # Must synchronize all GPUs.
            dist.barrier()
        
        dist.destroy_process_group()
        

if __name__ == '__main__':
    args = parse_args()

    dataset = PygNodePropPredDataset(name='ogbn-products', root='../dataset/')

    world_size = torch.cuda.device_count()

    print(f'Using {world_size} GPUs...')

    mp.spawn(run, args=(world_size, dataset, args), nprocs=world_size, join=True)       # join : Perform a blocking join on all processes.