"""
Edit by HappySky12

Official : https://github.com/pyg-team/pytorch_geometric/blob/master/examples/cluster_gcn_reddit.py

ClusterGCN seems much faster than GraphSAGE, but inference time seems much more slower than GraphSAGE.
"""

import argparse
import time

import torch
import torch.nn.functional as F

from torch_geometric.datasets import Reddit
from torch_geometric.loader import ClusterData, ClusterLoader, NeighborLoader
from torch_geometric.nn import ClusterGCNConv

parser = argparse.ArgumentParser(description='Reddit (ClusterGCN)')

parser.add_argument('--device', type=int, default=0)            # Specific device number. (0 to 3) 
parser.add_argument('--num_layers', type=int, default=3)        # Original ClusterGCN paper used 2~4.
parser.add_argument('--hidden_channels', type=int, default=128) # Original ClusterGCN paper used 128.
parser.add_argument('--dropout', type=float, default=0.2)       # Original ClusterGCN paper used 0.2.
parser.add_argument('--lr', type=float, default=0.01)           # Original ClusterGCN paper used 0.01.
parser.add_argument('--epochs', type=int, default=300)          
parser.add_argument('--num_parts', type=int, default=1500, help="# of partitions of entire graph. Partitioned sub-graphs will used for sub-graph sampling.")                                                                # Original ClusterGCN used 1500.
parser.add_argument('--batch_size', type=int, default=20, help="# of partitioned subgraphs in mini-batch. Each mini-batch will contain **batch_size** sub-graphs, and all sub-graphs in each batch will inter-connected.")      # Original ClusterGCN used 20.

args = parser.parse_args()

print(args)

device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

# We don't use T.ToSparseTensor(), because ClusterData makes assertion to sparse tensor. 
dataset = Reddit(root='../dataset/Reddit')

# Data(x=[232965, 602], edge_index=[2, 114615892], y=[232965], train_mask=[232965], val_mask=[232965], test_mask=[232965])
data = dataset[0]

partition_start_time = time.time()

# If we use 'save_dir' option, partitioned graph will be saved and can reuse it.
partitioned_graphs = ClusterData(data, num_parts=args.num_parts, recursive=False, save_dir=dataset.processed_dir)

partition_end_time = time.time()

print(f'Total partitioning time : {partition_end_time - partition_start_time:.4f}s') # takes about 130s.


# ClusterLoader will make mini-batches by using partitioned sub-graphs.
# And all sub-graphs in each mini-batch will inter-connected each other.
# So, we can see inter-connected sub-graphs in each mini-batch as a (little) full-graph. (So original ClusterGCN performs full-batch GCN in each mini-batch.)
train_loader = ClusterLoader(partitioned_graphs, batch_size=args.batch_size, shuffle=True, num_workers=12)

# For inference : this inference should be perfomed in entire graph.
subgraph_loader = NeighborLoader(data, input_nodes=None, num_neighbors=[-1], shuffle=False, batch_size=1024, num_workers=12, persistent_workers=True)

subgraph_loader.data.num_nodes = data.num_nodes
subgraph_loader.data.n_id = torch.arange(data.num_nodes)

class ClusterGCN(torch.nn.Module):
    """Cluster-GCN"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(ClusterGCN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        self.conv_layers = torch.nn.ModuleList()

        self.conv_layers.append(ClusterGCNConv(in_channels, hidden_channels, add_self_loops=True))
        for _ in range(self.num_layers - 2):
            self.conv_layers.append(ClusterGCNConv(hidden_channels, hidden_channels, add_self_loops=True))
        self.conv_layers.append(ClusterGCNConv(hidden_channels, out_channels, add_self_loops=True))
    
    def reset_parameters(self):
        for conv in self.conv_layers:
            conv.reset_parameters()
    
    def forward(self, x, edge_index):
        for conv in self.conv_layers[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv_layers[-1](x, edge_index)

        return x

    @torch.no_grad()
    def inference(self, x_all, subgraph_loader):
        for i, conv in enumerate(self.conv_layers):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(device)
                x = conv(x, batch.edge_index.to(device))
            
                if i < self.num_layers - 1:
                    x = F.relu(x)

                xs.append(x[:batch.batch_size].cpu())

            x_all = torch.cat(xs, dim=0)

        return x_all

model = ClusterGCN(
    in_channels=data.num_features,
    hidden_channels=args.hidden_channels,
    out_channels=dataset.num_classes,
    num_layers=args.num_layers,
    dropout=args.dropout
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

def train(epoch):
    model.train()

    total_loss = total_nodes = 0
    for batch in train_loader:
        batch = batch.to(device)                            # move each partitioned graphs in minibatch to device.

        optimizer.zero_grad()

        y = batch.y[batch.train_mask]               
        y_hat = model(batch.x, batch.edge_index)            # perform full-batch like propagation.

        loss = F.cross_entropy(y_hat[batch.train_mask], y)  # when calculating loss, we should correctly take train nodes from batch.

        loss.backward()
        optimizer.step()

        nodes_in_batch = batch.train_mask.sum().item()
        total_loss += loss.item() * nodes_in_batch
        total_nodes += nodes_in_batch

    return total_loss / total_nodes

@torch.no_grad()
def test():
    model.eval()

    y_hat = model.inference(data.x, subgraph_loader).argmax(dim=-1)
    y = data.y.to(y_hat.device)

    accuracy_list = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accuracy_list.append(int((y_hat[mask] == y[mask]).sum()) / int(mask.sum()))
    
    return accuracy_list

model.reset_parameters()

best_val_acc = best_test_acc = 0

start_time = torch.cuda.Event(enable_timing=True)
end_time = torch.cuda.Event(enable_timing=True)

for epoch in range(args.epochs):
    start_time.record()
    loss = train(epoch)
    end_time.record()

    torch.cuda.synchronize()
    train_elapsed_time = start_time.elapsed_time(end_time) / 1000.0

    start_time.record()
    train_acc, val_acc, tmp_test_acc = test()
    end_time.record()

    torch.cuda.synchronize()
    infer_elapsed_time = start_time.elapsed_time(end_time) / 1000.0

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc

    print(f'Epoch {epoch:03d}, ' 
    f'Loss: {loss:.4f}, '
    f'Train Acc: {train_acc:.4f}, '
    f'Val Acc: {val_acc:.4f}, '
    f'Test Acc: {tmp_test_acc:.4f}, '
    f'Train Time: {train_elapsed_time:.8f}s, '
    f'Infer Time: {infer_elapsed_time:.8f}s')

print(f'Best Val Acc:{best_val_acc:.4f}, Best Test Acc:{best_test_acc:.4f}')