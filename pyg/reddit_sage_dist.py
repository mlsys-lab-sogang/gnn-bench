"""
Data-parallel training script for GraphSAGE with Reddit dataset.
The original source from PyTorch Geometric is available at:
    https://github.com/pyg-team/pytorch_geometric/blob/master/examples/multi_gpu/distributed_sampling.py
"""
import argparse
import copy
import os

import pandas as pd

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from torch_geometric.datasets import Reddit
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Data-parallel training of GraphSAGE with Reddit")
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--fanout', type=int, nargs='+', help="# of fanouts.", required=True)
    parser.add_argument('--batch_size', type=int, help="Number of anchor nodes in each batch. The number of batches would be 'len(num_nodes)/len(batch_size)'", required=True)
    parser.add_argument('--num_nodes', type=int, default=-1, help="Number of available nodes/hosts.")           # All available nodes. Each node has `num_gpus` gpus.
    parser.add_argument('--node_id', type=int, default=0, help="Unique ID to identify the current node/host.")  # Current node id. If we have 2 nodes, each script's `node_id` will be 0 and 1.
    parser.add_argument('--num_gpus', type=int, default=-1, help="Number of GPUs in each node.")                # Number of GPUs in current node.

    args = parser.parse_args()

    assert len(args.fanout) == args.num_layers, \
        f"Length of fanout should be same with num_layers: len(fanout)({len(args.fanout)}) != num_layers({args.num_layers}))."

    return args


class SAGE_Dist(torch.nn.Module):
    """Mini-batch GraphSAGE"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(SAGE_Dist, self).__init__()

        self.conv_layers = torch.nn.ModuleList()
        self.conv_layers.append(SAGEConv(in_channels, hidden_channels, aggr='mean'))         # first layer
        for _ in range(num_layers - 2):
            self.conv_layers.append(SAGEConv(hidden_channels, hidden_channels, aggr='mean')) # hidden layers
        self.conv_layers.append(SAGEConv(hidden_channels, out_channels, aggr='mean'))        # last layer

        self.dropout = dropout
        self.num_layers = num_layers

    def reset_parameters(self):
        for conv in self.conv_layers:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for conv in self.conv_layers[:-1]:          # message passing from 1st layer to hidden layers
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv_layers[-1](x, edge_index)     # message passing from last hidden layer to last layer
        return x
    
    @torch.no_grad()
    def inference(self, x_all, device, subgraph_loader):
        # At each layer, GraphSAGE takes **all 1-hop neighbors** to compute node representations.
        # This leads to faster computation in contrast to immediately computing final representations of each batch.
        for i, conv in enumerate(self.conv_layers):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(device)   # access to original index, since index in batch is differ from original.    
                x = conv(x, batch.edge_index.to(device))            # message_passing for all nodes (in mini-batch).
                if i < len(self.conv_layers) - 1:
                    x = F.relu(x)
                xs.append(x[:batch.batch_size].cpu())               # will move it to main memory for inference.
            x_all = torch.cat(xs, dim=0)
        return x_all



# This function will copied to each GPU device.
# Process will made as # of GPUs, and each process will take each GPU.
# And each process will execute this function by using each GPU, in parallel manner.
# def run(rank, world_size, dataset, args):
def run(local_rank, dataset, args):
    """Run GraphSAGE in Distributed Data Parallel (DDP) Manner."""

    global_rank = args.node_id * args.num_gpus + local_rank # 0 * 4 + 0/1/2/3 => 0,1,2,3 || 1 * 4 + 0/1/2/3 => 4,5,6,7

    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]

    # world_size is same as # of GPUs (in single machine - multi GPU).
    dist.init_process_group(backend='nccl', rank=global_rank, world_size=args.world_size, init_method=f"tcp://{master_addr}:{master_port}")

    data = dataset[0]

    # Send node features and labels to device for faster access during sampling.
    # data = data.to(rank, 'x', 'y')
    data = data.to(local_rank, 'x', 'y')

    # Split training indices into chunks as 'world_size'. 
    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    # train_idx = train_idx.split(train_idx.size(0) // world_size)[rank]
    train_idx = train_idx.split(train_idx.size(0) // args.world_size)[local_rank]

    train_loader = NeighborLoader(
        data = data,
        input_nodes = train_idx,            # will make mini-batches in each GPU, with partitioned train nodes.
        num_neighbors = args.fanout,
        shuffle = True,
        drop_last = True,
        batch_size = args.batch_size,       # nodes in data[train_idx] is anchor nodes to make computation graph in each mini-batch, and # of anchor node in each mini-batch is same as 'batch_size'.
        num_workers = 6,
        persistent_workers = True
    )

    # Create 1-hop evaluation neighbor loader.
    # This loader will used in inference.
    # At each layer, GraphSAGE takes **all 1-hop neighbors** to compute node representations.
    # This leads to faster computation in contrast to immediately computing final representations of each batch.
    # Refer : Inductive Representation Learning on Large Graphs [Hamilton et al., 2017] (GraphSAGE)
    # if rank == 0:
    if local_rank == 0:
        """Define neighbor loader for inference in 1st (main) process"""
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
    # ).to(rank)
    ).to(local_rank)

    # and wrap it with DDP Module.
    # model = DistributedDataParallel(model, device_ids=[rank])
    model = DistributedDataParallel(model, device_ids=[local_rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # For time checking, we will save each GPU's train time to .csv file. (csv file will made as # of GPUs.)
    batch_history = pd.DataFrame(columns=['step', 'elapsed_time', 'mem_allocated'])
    acc_history = pd.DataFrame(columns=['epoch', 'train_acc', 'valid_acc', 'test_acc'])

    for epoch in range(args.epochs):
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        model.train()

        for batch in train_loader:
            start_time.record()
            optimizer.zero_grad()

            # before_alloc = torch.cuda.memory_allocated(rank)
            before_alloc = torch.cuda.memory_allocated(local_rank)

            y = batch.y[:batch.batch_size]
            # y_hat = model(batch.x, batch.edge_index.to(rank))[:batch.batch_size]
            y_hat = model(batch.x, batch.edge_index.to(local_rank))[:batch.batch_size]

            # after_alloc = torch.cuda.memory_allocated(rank)
            after_alloc = torch.cuda.memory_allocated(local_rank)

            loss = F.cross_entropy(y_hat, y)

            loss.backward()
            optimizer.step()

            end_time.record()
            torch.cuda.synchronize()

            elapsed_time = start_time.elapsed_time(end_time) / 1000.0
            mem_allocaated = (after_alloc - before_alloc)/1024.0/1024.0

            batch_history.loc[len(batch_history)] = [len(batch_history), elapsed_time, mem_allocaated]
        
        # print(f'Epoch: {epoch:03d}, GPU: {rank}, Loss: {loss:.4f}')
        print(f'Epoch: {epoch:03d}, GPU: {local_rank}, Loss: {loss:.4f}')

        # Must synchronize all GPUs.
        dist.barrier()

        # We evaluate on a single process for now.
        # if rank == 0:
        if local_rank == 0:
            start_time.record()

            model.eval()
            with torch.no_grad():
                # out = model.module.inference(data.x, rank, subgraph_loader)
                out = model.module.inference(data.x, local_rank, subgraph_loader)
            result = out.argmax(dim=-1) == data.y.to(out.device)

            train_acc = int(result[data.train_mask].sum()) / int(data.train_mask.sum())
            val_acc = int(result[data.val_mask].sum()) / int(data.val_mask.sum())
            test_acc = int(result[data.test_mask].sum()) / int(data.test_mask.sum())

            end_time.record()

            torch.cuda.synchronize()

            elapsed_time = start_time.elapsed_time(end_time) / 1000.0

            acc_history.loc[len(acc_history)] = [epoch, train_acc, val_acc, test_acc]

            # print(f'Epoch: {epoch:03d}, GPU: {rank}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}, Inference Time: {elapsed_time:.8f}s')
            print(f'Epoch: {epoch:03d}, GPU: {local_rank}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}, Inference Time: {elapsed_time:.8f}s')
        
        # synchronize all workers.
        dist.barrier()

    log_dir = '../logs/'

    if local_rank == 0:
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)
        acc_history.to_csv(os.path.join(log_dir, f'reddit_sage_dist_acc_layer{args.num_layers}_hidden{args.hidden_channels}_fanout{args.fanout}_batch{args.batch_size}.csv'), index=False)

    batch_history.to_csv(os.path.join(log_dir, f'reddit_sage_dist_layer{args.num_layers}_hidden{args.hidden_channels}_fanout{args.fanout}_batch{args.batch_size}_rank{local_rank}.csv'), index=False)

    dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()

    # FIXME: (0215) Since DDP cannot use SparseTensor, we use `edge_index`.
    # dataset = Reddit(root='../dataset/reddit/', transform=T.ToSparseTensor())
    data_dir = "../dataset/reddit/"
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    dataset = Reddit(root=data_dir)

    args.world_size = args.num_gpus * args.num_nodes # 4 * 2 = 8

    # world_size = torch.cuda.device_count()

    print(args)

    print(f"Using {args.world_size} GPUs")

    # mp.spawn(run, args=(world_size, dataset, args), nprocs=world_size, join=True) # join: exploits all processes in a blocking join manner.
    mp.spawn(run, args=(dataset, args), nprocs=args.num_gpus, join=True)    # join: exploits all processes in a blocking join manner.
