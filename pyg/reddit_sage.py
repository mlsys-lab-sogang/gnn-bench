"""
Edit by HappySky12 (2nd edit)
    - add some time-checking codes
    - save time results as .csv file to './time_results/' directory.

Official : https://github.com/pyg-team/pytorch_geometric/blob/master/examples/reddit.py (PyG example)

"""

import argparse
import copy
import os 

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T

from torch_geometric.datasets import Reddit
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader

from tqdm import tqdm

parser = argparse.ArgumentParser(description='Reddit (GraphSAGE)')

parser.add_argument('--device', type=int, default=0)            # Specific device number. (0 to 3) 
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--hidden_channels', type=int, default=128) 
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--train_type', help="'full'-batch vs. 'mini'-batch (via neighborhood sampling).", choices=['full', 'mini'], required=True)
parser.add_argument('--fanout', type=int, nargs='+', help="# of fanouts. Should be len(fanout) == len(num_layers).", required=False) 
parser.add_argument('--batch_size', type=int, help="# of anchor node in each batch. # of batch will be 'len(num_nodes)/len(batch_size)'", required=False)

args = parser.parse_args()
print(args)

if args.train_type == 'mini':
    if args.fanout is None or args.batch_size is None:
        raise Exception ("Should specify '--fanout' and '--batch_size'")

    if len(args.fanout) != args.num_layers:
        raise Exception (f"Fanout length should be same with 'num_layers' (len(fanout)({len(args.fanout)}) != num_layers({args.num_layers})).")

    print('mini-batch setting confirmed...')

device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

# Data(x=[232965, 602], y=[232965], train_mask=[232965], val_mask=[232965], test_mask=[232965], adj_t=[232965, 232965, nnz=114615892])
dataset = Reddit(root='../dataset/reddit/', transform=T.ToSparseTensor())

# If we are doing mini-batch manner, we should define NeighborLoader to sample neighbors.
if args.train_type == 'mini':
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Send node features and labels to GPU for faster access during sampling.
    start.record()
    data = dataset[0].to(device, 'x', 'y')
    end.record()

    torch.cuda.synchronize()

    print(data)
    
    # elapsed_time() returns milliseconds.
    print(f'Total GPU loading time : {start.elapsed_time(end) / 1000.0:.4f} s') # Total GPU loading time : 16.42 s

    train_loader = NeighborLoader(
        data = data,
        input_nodes = data.train_mask,      # will make mini-batches with train nodes.
        num_neighbors = args.fanout,
        shuffle = True,
        batch_size = args.batch_size,       # nodes in data[train_idx] is anchor nodes to make computation graph in each mini-batch, and # of anchor node in each mini-batch is same as 'batch_size'.
        num_workers = 6,
        persistent_workers = True
    )

    # Refer : Inductive Representation Learning on Large Graphs [Hamilton et al., 2017] (GraphSAGE)
    # This loader will used in inference.
    # At each layer, GraphSAGE takes **all 1-hop neighbors** to compute node representations.
    # This leads to faster computation in contrast to immediately computing final representations of each batch.
    subgraph_loader = NeighborLoader(
        data = copy.copy(data),
        input_nodes = None,                 # will make mini-batches with all nodes.
        num_neighbors = [-1],               # will consider all 1-hop neighbors to compute node representation.
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

else:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    data = dataset[0].to(device)
    end.record()

    torch.cuda.synchronize()

    print(data)
    print(f'Total GPU loading time : {start.elapsed_time(end) / 1000.0:.4f} s') # Total GPU loading time : 16.6524 s

class SAGE_Full(torch.nn.Module):
    """Full-batch GraphSAGE"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(SAGE_Full, self).__init__()

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

    def forward(self, x, adj_t):
        for conv in self.conv_layers[:-1]: # message passing from 1st layer to hidden layers
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv_layers[-1](x, adj_t) # message passing from last hidden layer to last layer
        return x


class SAGE_Mini(torch.nn.Module):
    """Mini-batch GraphSAGE"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(SAGE_Mini, self).__init__()

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
    def inference(self, x_all, subgraph_loader):
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

if args.train_type == 'mini':
    model = SAGE_Mini(
        in_channels = data.num_features,
        hidden_channels = args.hidden_channels,
        out_channels = dataset.num_classes,
        num_layers = args.num_layers,
        dropout = args.dropout
    )

else :
    model = SAGE_Full(
        in_channels = data.num_features,
        hidden_channels = args.hidden_channels,
        out_channels = dataset.num_classes,
        num_layers = args.num_layers,
        dropout = args.dropout        
    )

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

def train_full():
    """Full-batch train"""
    model.train()

    # progress_bar = tqdm(total=int(data.num_nodes))
    # progress_bar.set_description(f'Epoch {epoch:02d}')

    optimizer.zero_grad()

    # y_hat = model(data.x[data.train_mask], data.adj_t[data.train_mask].to(device))
    y_hat = model(data.x, data.adj_t)

    loss = F.cross_entropy(y_hat[data.train_mask], data.y[data.train_mask])

    loss.backward()
    optimizer.step()

    # progress_bar.close()

    return loss.item()

@torch.no_grad()
def test_full():
    """Full-batch test"""
    model.eval()

    y_hat = model(data.x, data.adj_t).argmax(dim=-1)

    accuracy_list = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accuracy_list.append(int((y_hat[mask] == data.y[mask]).sum()) / int(mask.sum()))

    return accuracy_list

def train_mini(epoch, dataframe:pd.DataFrame):
    """Mini-batch train"""
    model.train()

    # progress_bar = tqdm(total=int(len(train_loader.dataset)))
    # progress_bar.set_description(f'Epoch {epoch:02d}')

    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    total_loss = total_correct = total_examples = 0
    for batch in train_loader:
        num_nodes = batch.num_nodes - args.batch_size       # 'batch_size' is the same as '# of seed nodes' in each batch. so if we want to count **sampled nodes**, we can substract it.
        
        start_time.record()
        optimizer.zero_grad()
        
        before_alloc = torch.cuda.memory_allocated(device)

        y = batch.y[:batch.batch_size]
        y_hat = model(batch.x, batch.adj_t.to(device))[:batch.batch_size]

        after_alloc = torch.cuda.memory_allocated(device)

        loss = F.cross_entropy(y_hat, y)

        loss.backward()
        optimizer.step()

        end_time.record()
        torch.cuda.synchronize()

        elapsed_time = start_time.elapsed_time(end_time)/1000.0
        mem_allocated = (after_alloc - before_alloc)/1024.0/1024.0

        total_loss += float(loss) * batch.batch_size
        total_correct += int((y_hat.argmax(dim=-1) == y).sum())
        total_examples += batch.batch_size

        dataframe.loc[len(dataframe)] = [len(dataframe), elapsed_time, mem_allocated, num_nodes]
        # progress_bar.update(batch.batch_size)
    # progress_bar.close()

    return total_loss / total_examples, total_correct / total_examples, dataframe

@torch.no_grad()
def test_mini():
    """Mini-batch test"""
    model.eval()

    y_hat = model.inference(data.x, subgraph_loader).argmax(dim=-1)
    y = data.y.to(y_hat.device)    

    accuracy_list = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accuracy_list.append(int((y_hat[mask] == y[mask]).sum()) / int(mask.sum()))
    
    return accuracy_list

if args.train_type == 'mini':
    batch_history = pd.DataFrame(columns=['step', 'elapsed_time', 'mem_allocated', 'num_nodes'])
    acc_history = pd.DataFrame(columns=['epoch', 'elapsed_time_train', 'elapsed_time_infer', 'train_acc', 'valid_acc', 'test_acc'])

    model.reset_parameters()

    for epoch in range(args.epochs):
        gpu_start_time = torch.cuda.Event(enable_timing=True)
        gpu_end_time = torch.cuda.Event(enable_timing=True)

        gpu_start_time.record()
        loss, acc, batch_history = train_mini(epoch, batch_history)
        gpu_end_time.record()

        torch.cuda.synchronize()
        train_elapsed_time = gpu_start_time.elapsed_time(gpu_end_time) / 1000.0

        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Approx. Train Acc: {acc:.4f}, Training Time(GPU): {train_elapsed_time:.8f}s')

        gpu_start_time.record()
        train_acc, val_acc, test_acc = test_mini()
        gpu_end_time.record()

        torch.cuda.synchronize()
        infer_elapsed_time = gpu_start_time.elapsed_time(gpu_end_time) / 1000.0

        print(f'Epoch {epoch:03d}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}, Inference Time(GPU): {infer_elapsed_time:.8f} s')

        acc_history.loc[len(acc_history)] = [len(acc_history), train_elapsed_time, infer_elapsed_time, train_acc, val_acc, test_acc]

else:
    acc_history = pd.DataFrame(columns=['epoch', 'elapsed_time_train', 'elapsed_time_infer', 'train_acc', 'valid_acc', 'test_acc'])

    model.reset_parameters()

    for epoch in range(args.epochs):
        gpu_start_time = torch.cuda.Event(enable_timing=True)
        gpu_end_time = torch.cuda.Event(enable_timing=True)

        gpu_start_time.record()
        loss = train_full()
        gpu_end_time.record()        

        torch.cuda.synchronize()

        train_elapsed_time = gpu_start_time.elapsed_time(gpu_end_time) / 1000.0

        print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Training Time(GPU): {train_elapsed_time:.4f} s')

        gpu_start_time.record()
        train_acc, val_acc, test_acc = test_full()
        gpu_end_time.record()

        torch.cuda.synchronize()

        infer_elapsed_time = gpu_start_time.elapsed_time(gpu_end_time) / 1000.0

        print(f'Epoch {epoch:02d}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}, Inference Time(GPU): {infer_elapsed_time:.4f} s')

        acc_history.loc[len(acc_history)] = [len(acc_history), train_elapsed_time, infer_elapsed_time, train_acc, val_acc, test_acc]

dir_name = '../logs/single_gpu/'

if not os.path.isdir(dir_name):
    os.mkdir(dir_name)

if args.train_type == 'mini':
    batch_history.to_csv(dir_name + f'reddit_sage_layer{args.num_layers}_hidden{args.hidden_channels}_fanout{args.fanout}_batch{args.batch_size}_{args.train_type}.csv', index=False)
    acc_history.to_csv(dir_name + f'reddit_sage_acc_layer{args.num_layers}_hidden{args.hidden_channels}_fanout{args.fanout}_batch{args.batch_size}_{args.train_type}.csv', index=False)

else:
    acc_history.to_csv(dir_name + f'reddit_sage_acc_layer{args.num_layers}_hidden{args.hidden_channels}_{args.train_type}.csv', index=False)