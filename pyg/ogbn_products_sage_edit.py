"""
Edit by HappySky12

Official 1 : https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/products/gnn.py (OGB official examples)
Official 2 : https://github.com/pyg-team/pytorch_geometric/blob/master/examples/ogbn_products_sage.py (PyG official examples)

- This script will consume large amounts of GPU memory
""" 

import argparse
import copy
import os

# Since OGB stucks in last import, moved it to here.
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader

from logger import Logger

parser = argparse.ArgumentParser(description='OGBN-Products (GraphSAGE)')
parser.add_argument('--device', type=int, default=0)            # Specific device number. (0 to 3) 
parser.add_argument('--log_steps', type=int, default=1)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--hidden_channels', type=int, default=128) # 256 occurs CUDA OOM on full-batch 
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--train_type', help="'full'-batch vs. 'mini'-batch (via neighborhood sampling).", choices=['full', 'mini'], required=True)
parser.add_argument('--fanout', type=int, nargs='+', help="# of fanouts. Should be len(fanout) == len(num_layers).", required=False) 
parser.add_argument('--batch_size', type=int, help="# of anchor node in each batch. # of batch will be 'len(num_nodes)/len(batch_size)'", required=False)

args = parser.parse_args()

# For argument filtering & checking.
if args.train_type == 'mini':
    if args.fanout is None or args.batch_size is None:
        raise Exception ("Should specify '--fanout' and '--batch_size'")

    if len(args.fanout) != args.num_layers:
        raise Exception (f"Fanout length should be same with 'num_layers' (len(fanout)({len(args.fanout)}) != num_layers({args.num_layers})).")

    print('mini-batch setting confirmed...')

print(args)

device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

# https://pytorch-geometric.readthedocs.io/en/latest/notes/sparse_tensor.html (Difference between using 'edge_index' and 'adj_t'.)
dataset = PygNodePropPredDataset(name='ogbn-products', root='../dataset/', transform=T.ToSparseTensor())

# Because data is transformed to SparseTensor, we should use 'data.adj_t' (not 'data.edge_index') in computation.
# ogbn-products has adj matrix shaped by (2449029, 2449029), and # of non-zero values are 123718280.
# non-zero value's ratio is only 0.00206%.

# Data(num_nodes=2449029, x=[2449029, 100], y=[2449029, 1], adj_t=[2449029, 2449029, nnz=123718280])
data = dataset[0]

# OGB dataset has no mask. Instead they provides 'split_index()'.
# So we can split indexes and use them as mask index.
split_idx = dataset.get_idx_split()
train_idx = split_idx['train']      # will use only 'train' data in training.

# If we are doing mini-batch manner, we should define NeighborLoader to sample neighbors.
if args.train_type == 'mini':
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    data.y = data.y.squeeze()       # Since y's dim is (2449029, 1), reduce it to (2449029,)

    start.record()

    # Send node features and labels to GPU for faster access during sampling.
    data = data.to(device, 'x', 'y')
    end.record()

    torch.cuda.synchronize()

    print(f'Total GPU loading time : {start.elapsed_time(end) / 1000.0:.4f}s')

    train_loader = NeighborLoader(
        data = data, 
        input_nodes = train_idx,        # will make mini-batches with data[train_idx].
        num_neighbors = args.fanout,
        shuffle = True,
        batch_size = args.batch_size,   # nodes in data[train_idx] is anchor nodes to make computation graph in each mini-batch, and # of anchor node in each mini-batch is same as 'batch_size'.
        num_workers = 12,
        persistent_workers = True
    )

    # Refer : Inductive Representation Learning on Large Graphs [Hamilton et al., 2017] (GraphSAGE)
    # This loader is for inference.
    # At each layer, GraphSAGE takes "all 1-hop neighbors" to compute node representations.
    # This leads to faster computation in contrast to immediately computing final representations of each batch.
    subgraph_loader = NeighborLoader(
        data = copy.copy(data),
        input_nodes = None,             # will make mini-batches with all data
        num_neighbors = [-1],           # will consider all 1-hop neighbors to compute node representation
        shuffle = False,
        batch_size = args.batch_size,
        num_workers = 12,
        persistent_workers = True
    )

    # We don't need to maintain these features during evaluation, so delete it.
    del subgraph_loader.data.x, subgraph_loader.data.y

    # Add global node index information for mini-batch inference.
    subgraph_loader.data.num_nodes = data.num_nodes
    subgraph_loader.data.n_id = torch.arange(data.num_nodes)

else:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    data = data.to(device)
    end.record()

    torch.cuda.synchronize()

    print(f'Total GPU loading time : {start.elapsed_time(end) / 1000.0:.4f}s')


class SAGE_Full(torch.nn.Module):
    """Full-batch GraphSAGE"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(SAGE_Full, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr='mean'))          # 1st layer
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr='mean'))  # hidden layers
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr='mean'))         # last layer

        self.dropout = dropout
        self.num_layers = num_layers

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    # https://github.com/pyg-team/pytorch_geometric/discussions/4901 
    # adj is transposed into adj_t, it accelerates message passing.
    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:                                    # message passing from 1st layer to hidden layers
            x = conv(x, adj_t) 
            x = F.relu(x) 
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)                                    # message passing from last hidden layer to last layer
        return torch.log_softmax(x, dim=-1)                             # last layer activation (it's multi-label classification)


class SAGE_Mini(torch.nn.Module):
    """Mini-batch GraphSAGE"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(SAGE_Mini, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr='mean'))          # 1st layer
        for _ in range(num_layers -2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr='mean'))  # hidden layers
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr='mean'))         # last layer

        self.dropout = dropout
        self.num_layers = num_layers
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    
    # (0, Data(num_nodes=409009, x=[409009, 100], y=[409009, 1], adj_t=[409009, 409009, nnz=713999], input_id=[1024], batch_size=1024))
    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:                                    # message passing from 1st layer to hidden layers
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)                                    # message passing from last hidden layer to last layer
        return torch.log_softmax(x, dim=-1)

    @torch.no_grad()
    def inference(self, x_all, subgraph_loader):
        # At each layer, GraphSAGE takes "all 1-hop neighbors" to compute node representations.
        # This leads to faster computation in contrast to immediately computing final representations of each batch.
        for conv in self.convs[:-1]:
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(device)
                x = conv(x, batch.adj_t.to(device))
                x = F.relu(x)
                xs.append(x[:batch.batch_size])
            x_all = torch.cat(xs, dim=0)
        return x_all

if args.train_type == 'mini':
    model = SAGE_Mini(
        in_channels = dataset.num_features,
        hidden_channels = args.hidden_channels,
        out_channels = dataset.num_classes,
        num_layers = args.num_layers,
        dropout = args.dropout
    )

else:
    model = SAGE_Full(
        in_channels = dataset.num_features,
        hidden_channels = args.hidden_channels,
        out_channels = dataset.num_classes,
        num_layers = args.num_layers,
        dropout = args.dropout
    )

# Move model to GPU
model = model.to(device)

# pre-defined evaluator
evaluator = Evaluator(name='ogbn-products')
logger = Logger(args.runs, args)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

def train_full():
    """Full-batch train"""
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()

@torch.no_grad()
def test_full():
    model.eval()

    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc

def train_mini():
    """Mini-batch train"""
    model.train()

    total_loss = total_correct = total_examples = 0
    for batch in train_loader:
        optimizer.zero_grad()

        y = batch.y[:batch.batch_size]
        y_hat = model(batch.x, batch.adj_t.to(device))[:batch.batch_size]

        loss = F.nll_loss(y_hat, y)

        loss.backward()
        optimizer.step()

        total_loss += float(loss) * batch.batch_size
        total_correct += int((y_hat.argmax(dim=-1) == y).sum())
        total_examples += batch.batch_size

    return total_loss / total_examples, total_correct / total_examples

@torch.no_grad()
def test_mini():
    model.eval()

    out = model.inference(data.x, subgraph_loader)

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

    return train_acc, val_acc, test_acc


time_list_train, time_list_infer = np.array([]), np.array([])
loss_list_train = np.array([])
acc_list_train, acc_list_val, acc_list_test = np.array([]), np.array([]), np.array([])
if args.train_type == 'mini':
    for run in range(args.runs):
        model.reset_parameters()

        for epoch in range(args.epochs):
            time_start = torch.cuda.Event(enable_timing=True)
            time_end = torch.cuda.Event(enable_timing=True)

            time_start.record()
            loss, _ = train_mini()
            time_end.record()

            torch.cuda.synchronize()
            elapsed_time_train = time_start.elapsed_time(time_end) / 1000.0

            time_list_train = np.append(time_list_train, elapsed_time_train)
            loss_list_train = np.append(loss_list_train, loss)

            time_start.record()
            result = test_mini()
            time_end.record()

            torch.cuda.synchronize()
            elapsed_time_infer = time_start.elapsed_time(time_end) / 1000.0

            time_list_infer = np.append(time_list_infer, elapsed_time_infer)
            acc_list_train = np.append(acc_list_train, result[0])
            acc_list_val = np.append(acc_list_val, result[1])
            acc_list_test = np.append(acc_list_test, result[2])

            logger.add_result(run, result)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result

                print(f'Run: {run + 1:02d}, '
                    f'Epoch: {epoch:03d}, '
                    f'Loss: {loss:.4f}, '
                    f'Train: {100 * train_acc:.2f}%, '
                    f'Valid: {100 * valid_acc:.2f}%, '
                    f'Test: {100 * test_acc:.2f}%, '
                    f'Train Time: {elapsed_time_train:.8f}s,'
                    f'Infer Time: {elapsed_time_infer:.8f}s')
        print('====Mini-batch GraphSAGE result====')            
        logger.print_statistics(run)
    logger.print_statistics()

else:
    for run in range(args.runs):
        model.reset_parameters()

        for epoch in range(args.epochs):
            time_start = torch.cuda.Event(enable_timing=True)
            time_end = torch.cuda.Event(enable_timing=True)

            # time_start.record()
            loss = train_full()
            time_end.record()

            torch.cuda.synchronize()
            elapsed_time_train = time_start.elapsed_time(time_end) / 1000.0

            time_list_train = np.append(time_list_train, elapsed_time_train)
            loss_list_train = np.append(loss_list_train, loss)

            time_start.record()
            result = test_full()
            time_end.record()

            torch.cuda.synchronize()
            elapsed_time_infer = time_start.elapsed_time(time_end) / 1000.0

            time_list_infer = np.append(time_list_infer, elapsed_time_infer)
            acc_list_train = np.append(acc_list_train, result[0])
            acc_list_val = np.append(acc_list_val, result[1])
            acc_list_test = np.append(acc_list_test, result[2])

            logger.add_result(run, result)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result

                print(f'Run: {run + 1:02d}, '
                    f'Epoch: {epoch:03d}, '
                    f'Loss: {loss:.4f}, '
                    f'Train: {100 * train_acc:.2f}%, '
                    f'Valid: {100 * valid_acc:.2f}% '
                    f'Test: {100 * test_acc:.2f}%,'
                    f'Train Time: {elapsed_time_train:.8f}s,'
                    f'Infer Time: {elapsed_time_infer:.8f}s')
        print('====Full-batch GraphSAGE result====')            
        logger.print_statistics(run)
    logger.print_statistics()

df = pd.DataFrame({
    'loss' : loss_list_train,
    'train_time' : time_list_train,
    'acc_train' : acc_list_train,
    'acc_val' : acc_list_val,
    'acc_test' : acc_list_test,
    'infer_time' : time_list_infer
})

dir_name = './time_result/'

if args.train_type == 'mini':
    file_name = f'ogbn_products_sage_singleGPU_{args.train_type}_layer{args.num_layers}_hidden{args.hidden_channels}_fanout{args.fanout}_batch{args.batch_size}.csv'

else:
    file_name = f'ogbn_products_sage_singleGPU_{args.train_type}_layer{args.num_layers}_hidden{args.hidden_channels}.csv'

if not os.path.isdir(dir_name):
    os.mkdir(dir_name)

df.to_csv(dir_name + file_name)