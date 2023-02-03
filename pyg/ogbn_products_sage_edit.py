"""
Edit by HappySky12

Official 1 : https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/products/gnn.py (OGB official examples)
Official 2 : https://github.com/pyg-team/pytorch_geometric/blob/master/examples/ogbn_products_sage.py (PyG official examples)

- Full-batch train
- This script will consume large amounts of GPU memory
""" 

import argparse

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
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=128) # 256 occurs CUDA OOM on full-batch 
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--train_type', help="'full'-batch vs. 'mini'-batch (via neighborhood sampling).", choices=['full', 'mini'], required=True)
    parser.add_argument('--fanout', help="# of fanouts. Should be len(fanout) == len(num_layers).", required=False) 
    parser.add_argument('--batch_size', help="# of anchor node in each batch. # of batch will be 'len(num_nodes)/len(batch_size)'", required=False)

    args = parser.parse_args()
    print(args)

    # For argument filtering & checking.
    if args.train_type == 'mini':
        if args.fanout is None or args.batch_size is None:
            raise Exception ("Should specify '--fanout' and '--batch_size'")

        fanout_info = eval(args.fanout)

        if len(fanout_info) != args.num_layers:
            raise Exception (f"Fanout length should be same with 'num_layers' (len(fanout)({len(fanout_info)}) != num_layers({args.num_layers})).")

        print('mini-batch setting confirmed...')

    # quit()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    dataset = PygNodePropPredDataset(name='ogbn-products', root='../dataset/', transform=T.ToSparseTensor())

    
    # Because data is transformed to SparseTensor, we should use 'data.adj_t' (not 'data.edge_index').
    # ogbn-products has adj matrix shaped by (2449029, 2449029), and # of non-zero values are 123718280.
    # non-zero value's ratio is only 0.00206%.

    # Data(num_nodes=2449029, x=[2449029, 100], y=[2449029, 1], adj_t=[2449029, 2449029, nnz=123718280])
    data = dataset[0]

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device) # will use only 'train' data in training, so send data's train idx info to GPU.

    # TODO: Full-batch vs. Mini-batch handling
    # TODO: For mini-batch, we should specify "NeighborLoader" to do neighborhood sampling.
    if args.train_type == 'mini':
        model = SAGE_Mini(
            in_channels = data.num_features,
            hidden_channels = args.hidden_channels, 
            out_channels = dataset.num_classes,
            num_layers = args.num_layers,
            dropout = args.dropout
        )
    
    else:
        model = SAGE_Full(
            in_channels = data.num_features, 
            hidden_channels = args.hidden_channels, 
            out_channels = dataset.num_classes,
            num_layers = args.num_layers,
            dropout = args.dropout
        )

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
