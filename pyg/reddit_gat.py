"""
Edit by HappySky12

Official : https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gat.py (uses Planetoid dataset.)

GAT seems much more slower than GraphSAGE, at least in my desktop.
"""
import argparse
import copy

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T

from torch_geometric.datasets import Reddit
from torch_geometric.nn import GATConv
from torch_geometric.loader import NeighborLoader

from tqdm.auto import tqdm

parser = argparse.ArgumentParser(description='Reddit (GAT)')

parser.add_argument('--device', type=int, default=0)            # Specific device number. (0 to 3) 
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--hidden_channels', type=int, default=128) 
parser.add_argument('--dropout', type=float, default=0.6)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--fanout', type=int, nargs='+', help="# of fanouts. Should be len(fanout) == len(num_layers).", required=False) 
parser.add_argument('--batch_size', type=int, help="# of anchor node in each batch. # of batch will be 'len(num_nodes)/len(batch_size)'", required=False)
parser.add_argument('--heads', type=int, default=2, help="# of multi-head attention heads. In GAT, it means # of attention computation's iteration.")

args = parser.parse_args()

if args.fanout is None or args.batch_size is None:
    raise Exception ("Should specify '--fanout' and '--batch_size'")

if len(args.fanout) != args.num_layers:
    raise Exception (f"Fanout length should be same with 'num_layers' (len(fanout)({len(args.fanout)}) != num_layers({args.num_layers})).")

print(args)

device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

# Data(x=[232965, 602], y=[232965], train_mask=[232965], val_mask=[232965], test_mask=[232965], adj_t=[232965, 232965, nnz=114615892])
dataset = Reddit(root='../dataset/reddit', transform=T.ToSparseTensor())

data = dataset[0].to(device, 'x','y')

print('Data loading done...')

train_loader = NeighborLoader(
    data=data,
    input_nodes=data.train_mask,
    num_neighbors=args.fanout,
    shuffle=True,
    batch_size=args.batch_size,
    num_workers=6,
    persistent_workers=True
)

subgraph_loader = NeighborLoader(
    data=copy.copy(data),
    input_nodes=None,
    num_neighbors=[-1],
    shuffle=False,
    batch_size=args.batch_size,
    num_workers=6,
    persistent_workers=True
)

del subgraph_loader.data.x, subgraph_loader.data.y

subgraph_loader.data.num_nodes = data.num_nodes
subgraph_loader.data.n_id = torch.arange(data.num_nodes)

# [Paper] Transdictive는 2 layer, K=8 // Inductive는 3 layer, 앞 2개는 K=4, 마지막 1개는 K=6 + skip connection
class GAT(torch.nn.Module):
    """Mini-batch GAT"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, heads, dropout):
        super(GAT, self).__init__()

        self.conv_layers = torch.nn.ModuleList()

        self.conv_layers.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.conv_layers.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))
        self.conv_layers.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout))     # At last layer, do not concat feature matrix, average it. 

        self.num_layers = num_layers
        self.dropout = dropout 

    def reset_parameters(self):
        for conv in self.conv_layers:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.conv_layers[:-1]:
            x = conv(x, adj_t)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv_layers[-1](x, adj_t)

        return x

    @torch.no_grad()
    def inference(self, x_all, subgraph_loader):
        progress_bar = tqdm(total=int(len(subgraph_loader.dataset) * len(self.conv_layers)))
        progress_bar.set_description('Evaluating...')

        # At each layer, takes **all 1-hop neighbors** to compute node representations.
        # This leads to faster computation in contrast to immediately computing final representations of each batch.
        for i, conv in enumerate(self.conv_layers):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(device)
                x = conv(x, batch.adj_t.to(device))

                if i < len(self.conv_layers) - 1:
                    x = F.elu(x)
                
                xs.append(x[:batch.batch_size].cpu())

                progress_bar.update(batch.batch_size)

            x_all = torch.cat(xs, dim=0)
        
        progress_bar.close()
        
        return x_all

model = GAT(
    in_channels=data.num_features,
    hidden_channels=args.hidden_channels,
    out_channels=dataset.num_classes,
    num_layers=args.num_layers,
    dropout=args.dropout,
    heads=args.heads                                 # 8 occurs OOM, most of implementations (PyG, OGB leaderboard) uses 2~4 heads.
).to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

def train(epoch):
    model.train()

    progress_bar = tqdm(total=int(len(train_loader.dataset)))
    progress_bar.set_description(f'Epoch {epoch:03d}')
    # total_loss = total_correct = total_examples = 0
    total_loss = total_examples = 0
    for batch in train_loader:
        optimizer.zero_grad()

        y = batch.y[:batch.batch_size]
        y_hat = model(batch.x, batch.adj_t.to(device))[:batch.batch_size]

        loss = F.cross_entropy(y_hat, y)

        loss.backward()
        optimizer.step()

        total_loss += float(loss) * batch.batch_size
        # total_correct += int((y_hat.argmax(dim=-1) == y).sum())
        total_examples += batch.batch_size
        progress_bar.update(batch.batch_size)
    progress_bar.close()

    return total_loss / total_examples #, total_correct / total_examples

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

for epoch in range(300):
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
        best_test_acc = tmp_test_acc
    
    print(f'Epoch {epoch:03d}, ' 
    f'Loss: {loss:.4f}, '
    f'Train Acc: {train_acc:.4f}, '
    f'Val Acc: {val_acc:.4f}, '
    f'Test Acc: {tmp_test_acc:.4f}, '
    f'Train Time: {train_elapsed_time:.8f}s, '
    f'Infer Time: {infer_elapsed_time:.8f}s')

print(f'Best Val Acc:{best_val_acc:.4f}, Best Test Acc:{best_test_acc:.4f}')

