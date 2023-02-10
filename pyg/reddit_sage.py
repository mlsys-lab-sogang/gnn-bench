"""
Edit by HappySky12

Official : https://github.com/pyg-team/pytorch_geometric/blob/master/examples/reddit.py (PyG example)

TODO: 
    1. Mini-batch train (follow by official)
    2. Full-batch train
"""

import argparse
import copy

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
dataset = Reddit(root='../dataset/Reddit/', transform=T.ToSparseTensor())

## FIXME: full, mini 차이위해서 잠시 주석. (train_type 따라서 올리는거 다르게.)
## Send node features and labels to GPU for faster access during sampling.
# data = dataset[0].to(device, 'x', 'y')


# If we are doing mini-batch manner, we should define NeighborLoader to sample neighbors.
if args.train_type == 'mini':
    data = dataset[0].to(device, 'x', 'y')
    print(data)

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
    data = dataset[0].to(device)
    print(data)


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
        progress_bar = tqdm(total=len(subgraph_loader.dataset) * len(self.conv_layers))
        progress_bar.set_description('Evaluating...')

        """
        for conv in self.conv_layers[:-1]:
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(device)   
                # print(batch.n_id.shape)
                # print(x_all.shape)
                # print(x.shape)
                # quit()
                x = conv(x, batch.adj_t.to(device))                 
                x = F.relu(x)
                xs.append(x[:batch.batch_size].cpu())               
                progress_bar.update(batch.batch_size)
            x_all = torch.cat(xs, dim=0)
        progress_bar.close()
        return x_all
        """

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
                progress_bar.update(batch.batch_size)
            x_all = torch.cat(xs, dim=0)
        progress_bar.close()
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

def train_full(epoch):
    """Full-batch train"""
    model.train()

    progress_bar = tqdm(total=int(data.num_nodes))
    progress_bar.set_description(f'Epoch {epoch:02d}')

    optimizer.zero_grad()

    # y_hat = model(data.x[data.train_mask], data.adj_t[data.train_mask].to(device))
    y_hat = model(data.x, data.adj_t)

    loss = F.cross_entropy(y_hat[data.train_mask], data.y[data.train_mask])

    loss.backward()
    optimizer.step()

    progress_bar.close()

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

def train_mini(epoch):
    """Mini-batch train"""
    model.train()

    progress_bar = tqdm(total=int(len(train_loader.dataset)))
    progress_bar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = total_examples = 0
    for batch in train_loader:
        optimizer.zero_grad()
        
        y = batch.y[:batch.batch_size]
        y_hat = model(batch.x, batch.adj_t.to(device))[:batch.batch_size]

        loss = F.cross_entropy(y_hat, y)

        loss.backward()
        optimizer.step()

        total_loss += float(loss) * batch.batch_size
        total_correct += int((y_hat.argmax(dim=-1) == y).sum())
        total_examples += batch.batch_size
        progress_bar.update(batch.batch_size)
    progress_bar.close()

    return total_loss / total_examples, total_correct / total_examples

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
    model.reset_parameters()

    for epoch in range(args.epochs):
        loss, acc = train_mini(epoch)
        print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train Acc: {acc:.4f}')

        train_acc, val_acc, test_acc = test_mini()
        print(f'Epoch: {epoch:02d}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')

else:
    model.reset_parameters()

    for epoch in range(args.epochs):
        loss = train_full(epoch)

        print(f'Epoch {epoch:02d}, Loss: {loss:.4f}')

        train_acc, val_acc, test_acc = test_full()
        print(f'Epoch: {epoch:02d}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')