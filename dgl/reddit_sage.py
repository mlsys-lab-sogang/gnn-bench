# https://github.com/dglai/dgl-0.5-benchmark/blob/master/end_to_end/sampling/node-classification/reddit/ns-sage-dgl.py
# https://github.com/dmlc/dgl/blob/master/examples/pytorch/graphsage/node_classification.py

import argparse
import copy
import os

import pandas as pd

import torch
import torch.nn.functional as F
import torchmetrics.functional as MF

from tqdm.auto import tqdm

from dgl.nn import SAGEConv
from dgl.data import RedditDataset
from dgl.dataloading import DataLoader, MultiLayerFullNeighborSampler, NeighborSampler

os.environ['DGLBACKEND'] = 'pytorch'

parser = argparse.ArgumentParser(description="Reddit (GraphSAGE)")

parser.add_argument('--device', type=int, default=0)
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

print(args)

# Since dgl's processed data output (.bin, .npz) is different from PyG (.pt), set another directory to save dataset.
if not os.path.isdir('./dataset/'):
    os.mkdir('./dataset')

device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

dataset = RedditDataset(raw_dir='./dataset/')

# Graph(num_nodes=232965, num_edges=114615892,
#      ndata_schemes={'label': Scheme(shape=(), dtype=torch.int64), 'feat': Scheme(shape=(602,), dtype=torch.float32), 'test_mask': Scheme(shape=(), dtype=torch.bool), 'val_mask': Scheme(shape=(), dtype=torch.bool), 'train_mask': Scheme(shape=(), dtype=torch.bool)}
#      edata_schemes={'__orig__': Scheme(shape=(), dtype=torch.int64)})
# check with `data.formats()`, data is made as coo matrix.
data = dataset[0]
data = data.to(device)

# Get total number of classes.
num_classes = dataset.num_classes

# Get node feature
features = data.ndata['feat']

# Get data split
train_mask, val_mask, test_mask = data.ndata['train_mask'], data.ndata['val_mask'], data.ndata['test_mask']

# Get data idx
train_idx = torch.nonzero(train_mask, as_tuple=True)[0].to(device)
val_idx = torch.nonzero(val_mask, as_tuple=True)[0].to(device)
test_idx = torch.nonzero(test_mask, as_tuple=True)[0]

# Get labels
label = data.ndata['label']

# Define NeighborSampler & DataLoader for mini-batch training
sampler = NeighborSampler(
    fanouts = args.fanout,
    prefetch_node_feats = ['feat'],
    prefetch_labels = ['label']
)

train_loader = DataLoader(
    graph = data,
    indices = train_idx,
    graph_sampler = sampler,
    device = args.device,
    batch_size = args.batch_size,
    shuffle = True,
    num_workers = 0
)

val_loader = DataLoader(
    graph = data,
    indices = val_idx,
    graph_sampler = sampler,
    device = args.device,
    batch_size = args.batch_size,
    shuffle = True,
    num_workers = 0
)

class SAGE(torch.nn.Module):
    """Mini-batch GraphSAGE"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(SAGE, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_size = hidden_channels
        self.output_size = out_channels

        self.conv_layers = torch.nn.ModuleList()

        self.conv_layers.append(SAGEConv(in_channels, hidden_channels, aggregator_type='mean'))
        for _ in range(self.num_layers - 2):
            self.conv_layers.append(SAGEConv(hidden_channels, hidden_channels, aggregator_type='mean'))
        self.conv_layers.append(SAGEConv(hidden_channels, out_channels, aggregator_type='mean'))

    # dataloader will return blocks
    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.conv_layers, blocks)):
            h = layer(block, h)                 # message passing between layers
            if l != len(self.conv_layers) - 1:  # apply dropout & non-linear in hidden layers
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h
    
    def inference(self, data, device, batch_size):
        # Take **all 1-hop neighbors** to compute node representations for each layer.
        # (layer-wise inference to get all node embeddings.)
        features = data.ndata['feat']

        # Define 1-hop full neighbor sampler & loader.
        sampler = MultiLayerFullNeighborSampler(
            num_layers=1,
            prefetch_node_feats=['feat']
        )
        dataloader = DataLoader(
            graph = data,
            indices = torch.arange(data.num_nodes()).to(data.device),
            graph_sampler = sampler,
            device = device,
            batch_size = batch_size,
            shuffle = False,
            drop_last = False,
            num_workers = 0
        )

        # TODO: ?
        buffer_device = torch.device('cpu')
        pin_memory = buffer_device != device

        for l, layer in enumerate(self.conv_layers):
            # dummy tensor for calculating node representations.
            y = torch.empty(
                data.num_nodes(),
                self.hidden_size if l != len(self.conv_layers) - 1 else self.output_size,
                device = buffer_device,
                pin_memory = pin_memory
            )

            features = features.to(device)

            # calculate node representations.
            for input_nodes, output_nodes, blocks in tqdm(dataloader):
                x = features[input_nodes]
                x = layer(blocks[0], x)
                if l != len(self.conv_layers) - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout)
                # by design, output nodes are contiguous.
                y[output_nodes[0] : output_nodes[-1] + 1] = x.to(buffer_device)
            
            features = y
        
        return y

model = SAGE(
    in_channels = data.ndata['feat'].shape[1],
    hidden_channels = args.hidden_channels,
    out_channels = dataset.num_classes,
    num_layers = args.num_layers,
    dropout = args.dropout
)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

@torch.no_grad()
def evaluate(model, data, dataloader):
    model.eval()

    ys = []
    y_hats = []

    for _, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        x = blocks[0].srcdata['feat']
        ys.append(blocks[-1].dstdata['label'])
        y_hats.append(model(blocks, x))
    
    return MF.accuracy(preds=torch.cat(y_hats), target=torch.cat(ys))

@torch.no_grad()
def layerwise_infer(device, data, node_idx, model, batch_size):
    model.eval()

    pred = model.inference(data, device, batch_size)
    pred = pred[node_idx]
    label = data.ndata['label'][node_idx].to(pred.device)

    return MF.accuracy(preds=pred, target=label)

def train(args, data, model):
    for epoch in range(args.epochs):
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        model.train()

        total_loss = 0

        start_time.record()
        # blocks = [Block(3hop-2hop), Block(2hop-1hop), Block(1hop-anchor)] 
        for iter, (input_nodes, output_nodes, blocks) in enumerate(train_loader):
            optimizer.zero_grad()

            x = blocks[0].srcdata['feat']
            y = blocks[-1].dstdata['label']

            y_hat = model(blocks, x)

            loss = F.cross_entropy(y_hat, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        accuracy = evaluate(model, data, val_loader)
        end_time.record()
        torch.cuda.synchronize()

        elapsed_time = start_time.elapsed_time(end_time)/1000.0
        print(f'Epoch {epoch:03d}, Loss: {total_loss / (iter + 1):.4f}, Val_Acc: {accuracy.item():.4f}, Training Time: {elapsed_time:.8f}s')

train(args, data, model)
acc = layerwise_infer(device, data, test_idx, model, batch_size=4096)
print(f'Test Acc: {acc.item():.4f}')