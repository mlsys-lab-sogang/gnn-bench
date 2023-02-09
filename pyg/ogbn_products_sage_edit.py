"""
Edit by HappySky12

Official 1 : https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/products/gnn.py (OGB official examples)
Official 2 : https://github.com/pyg-team/pytorch_geometric/blob/master/examples/ogbn_products_sage.py (PyG official examples)

- Full-batch train
- This script will consume large amounts of GPU memory
""" 

import argparse
import copy

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from logger import Logger

class SAGE_Full(torch.nn.Module):
    """Full-batch GraphSAGE"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(SAGE_Full, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr='mean')) # 1st layer
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr='mean')) # hidden layers
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr='mean')) # last layer

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    # TODO: ToSparseTensor로 transform없이 edge_index로 할때 차이? 단순히 정보량?
    # https://github.com/pyg-team/pytorch_geometric/discussions/4901 
    # adj is transposed into adj_t, it accelerates message passing.
    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t) # message passing
            x = F.relu(x) # activation
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t) # last layer message passing
        return torch.log_softmax(x, dim=-1) # last layer activation (multi-label classification)


class SAGE_Mini(torch.nn.Module):
    """Mini-batch GraphSAGE"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(SAGE_Mini, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr='mean')) # 1st layer
        for _ in range(num_layers -2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr='mean')) # hidden layers
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr='mean')) # last layer

        self.dropout = dropout
        self.num_layers = num_layers
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    
    # TODO: forward, inference 작성
    # (0, Data(num_nodes=409009, x=[409009, 100], y=[409009, 1], adj_t=[409009, 409009, nnz=713999], input_id=[1024], batch_size=1024))
    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t) # message passing
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
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

 

# FIXME: train, test 함수는 1개로 -> 그 안에서 args.train_type 에 따라 if-else 나눠서 train-test 분리 : 받는 인자에 argument도 받도록.
# train, test가 받아야할 인자 : x, adj_t, train_type, loader

def train_full(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()

# TODO: train_mini 작성


@torch.no_grad()
def test(model, data, split_idx, evaluator):
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


def main():
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
    print(args)

    # For argument filtering & checking.
    if args.train_type == 'mini':
        if args.fanout is None or args.batch_size is None:
            raise Exception ("Should specify '--fanout' and '--batch_size'")

        if len(args.fanout) != args.num_layers:
            raise Exception (f"Fanout length should be same with 'num_layers' (len(fanout)({len(args.fanout)}) != num_layers({args.num_layers})).")

        print('mini-batch setting confirmed...')

    # quit()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    # https://pytorch-geometric.readthedocs.io/en/latest/notes/sparse_tensor.html (Difference between using 'edge_index' and 'adj_t'.)
    dataset = PygNodePropPredDataset(name='ogbn-products', root='../dataset/', transform=T.ToSparseTensor())

    
    # Because data is transformed to SparseTensor, we should use 'data.adj_t' (not 'data.edge_index') in computation.
    # ogbn-products has adj matrix shaped by (2449029, 2449029), and # of non-zero values are 123718280.
    # non-zero value's ratio is only 0.00206%.

    # Data(num_nodes=2449029, x=[2449029, 100], y=[2449029, 1], adj_t=[2449029, 2449029, nnz=123718280])
    data = dataset[0]

    split_idx = dataset.get_idx_split()

    # FIXME: .to(device) 한 후에 NeighborLoader 호출시 CUDA initialization error. data와 idx가 다른 
    train_idx = split_idx['train']#.to(device) # will use only 'train' data in training, so send data's train idx info to GPU.
    
    if args.train_type == 'mini':
        model = SAGE_Mini(
            in_channels = data.num_features,
            hidden_channels = args.hidden_channels, 
            out_channels = dataset.num_classes,
            num_layers = args.num_layers,
            dropout = args.dropout
        )

        train_loader = NeighborLoader(
            data = data, 
            input_nodes = train_idx,        # will make mini-batches with data[train_idx].
            num_neighbors = args.fanout,
            shuffle = True,
            batch_size = args.batch_size,   # nodes in data[train_idx] is anchor nodes to make computation graph in each mini-batch, and # of anchor node in each mini-batch is same as 'batch_size'.
            num_workers = 6,
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
            num_workers = 6,
            persistent_workers = True
        )

        # add global node index information for mini-batch inference.
        subgraph_loader.data.n_id = torch.arange(data.num_nodes)
    
    else:
        model = SAGE_Full(
            in_channels = data.num_features, 
            hidden_channels = args.hidden_channels, 
            out_channels = dataset.num_classes,
            num_layers = args.num_layers,
            dropout = args.dropout
        )
    print(model)
    # print(train_loader)
    # print(next(enumerate(train_loader))) # (0, Data(num_nodes=409009, x=[409009, 100], y=[409009, 1], adj_t=[409009, 409009, nnz=713999], input_id=[1024], batch_size=1024))
    # print(type(next(enumerate(train_loader))))
    print(subgraph_loader)
    print(next(enumerate(subgraph_loader)))
    quit()

    # Move model & data to GPU
    model = model.to(device)
    data = data.to(device)

    evaluator = Evaluator(name='ogbn-products')
    logger = Logger(args.runs, args)

    # TODO: mini-batch run 작성
    if args.train_type == 'mini':
        print('mini')
    
    else:
        print('====Full-batch====')
        for run in range(args.runs):
            model.reset_parameters()

            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

            for epoch in range(1, 1 + args.epochs):
                loss = train_full(model, data, train_idx, optimizer)
                result = test(model, data, split_idx, evaluator)
                logger.add_result(run, result)

                if epoch % args.log_steps == 0:
                    train_acc, valid_acc, test_acc = result

                    print(f'Run: {run + 1:02d}, '
                        f'Epoch: {epoch:02d}, '
                        f'Loss: {loss:.4f}, '
                        f'Train: {100 * train_acc:.2f}%, '
                        f'Valid: {100 * valid_acc:.2f}% '
                        f'Test: {100 * test_acc:.2f}%')
            print('====Full-batch GraphSAGE result====')            
            logger.print_statistics(run)
        logger.print_statistics()


if __name__ == '__main__':
    main()
