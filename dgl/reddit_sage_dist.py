r"""
Data-parallel training script for GraphSAGE with Reddit dataset.
The original source from DGL is available at:
    https://github.com/dmlc/dgl/tree/master/examples/pytorch/graphsage/dist
"""
import argparse
import socket
import os
import logging

from contextlib import contextmanager
from datetime import datetime
from tqdm.auto import tqdm

import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

import dgl
from dgl.nn.pytorch import SAGEConv
from dgl.dataloading import NeighborSampler, DistNodeDataLoader

def parse_args():
    parser = argparse.ArgumentParser(description="Data-parallel training of GraphSAGE with Reddit")

    # args for distributed setting.
    parser.add_argument('--part_id', type=int, help="Graph partition id to use.")
    parser.add_argument('--ip_config', type=str, help="File(.txt) for IP configuration. File should have **all** participating cluster's IP (& Port) address.")
    parser.add_argument('--part_config', type=str, help="Path of partition config file(.json).")
    parser.add_argument('--standalone', action='store_true', help="Run in standalone mode. Usually used for testing.")
    parser.add_argument('--nnodes', type=int, default=2, help="Total number of machines participating in distributed training.")
    parser.add_argument('--nprocs', type=int, default=4, help="Total number of GPUs in current machine.")
    parser.add_argument('--node_id', type=int, required=True, help="Current machine's ID for distributed setting. (0 ~ 'num_cluster - 1')")

    # args for train setting.
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--fanout', type=int, nargs='+', required=True)

def load_subtensor(data, seeds, input_nodes, device, load_feature=True):
    r"""Copy features and labels of a set of nodes, send it to GPU"""
    batch_inputs = (data.ndata['features'][input_nodes].to(device) if load_feature else None)
    batch_labels = data.ndata['labels'][seeds].to(device)

    return batch_inputs, batch_labels

class DistSAGE(torch.nn.Module):
    r"Mini-batch GraphSAGE"
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(DistSAGE, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_size = hidden_channels
        self.output_size = out_channels

        self.conv_layers = torch.nn.ModuleList()
        
        self.conv_layers.append(SAGEConv(in_channels, hidden_channels, aggregator_type='mean'))
        for _ in range(self.num_layers - 2):
            self.conv_layers.append(SAGEConv(hidden_channels, hidden_channels, aggregator_type='mean'))
        self.conv_layers.append(SAGEConv(hidden_channels, out_channels, aggregator_type='mean'))
    
    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.conv_layers, blocks)):
            h = layer(block, h)                 # message passing between layers
            if l != len(self.conv_layers) - 1:  # apply non-linear & dropout in hidden layers
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h
    
    def inference(self, data:dgl.distributed.DistGraph, x, device, batch_size):
        r"""
        Distributed layer-wise inference.

        Take **all 1-hop neighbors** to compute nore representations for each layer.
        (layer-wise inference to get all node embeddings.)
        """
        features = data.ndata['feat']

        # split input nodes based on 'partition book', and return a subset of nodes for local rank.
        # more details : https://docs.dgl.ai/generated/dgl.distributed.node_split.html#dgl.distributed.node_split 
        node_ids = dgl.distributed.node_split(np.arange(data.num_nodes()), data.get_partition_book(), force_even=True)

        # access to distributed tensor, shareded and stored in machines.
        # more details : https://docs.dgl.ai/api/python/dgl.distributed.html#distributed-tensor
        y = dgl.distributed.DistTensor(shape=(data.num_nodes(), self.hidden_size), dtype=torch.float32, name='h', persistent=True)

        for i, layer in enumerate(self.conv_layers):
            if i == len(self.conv_layers) -1:
                y = dgl.distributed.DistTensor(shape=(data.num_nodes(), self.output_size), dtype=torch.float32, name='h_last', persistent=True)
            
            print(f'|V| = {data.num_nodes()}')

            sampler = NeighborSampler(fanouts=[-1])
            dataloader = DistNodeDataLoader(
                g = data,
                nids = node_ids,
                sampler = sampler,
                batch_size = batch_size,
                shuffle = False,
                drop_last = False
            )

            for input_nodes, output_nodes, blocks in tqdm(dataloader):
                block = blocks[0].to(device)

                # fetch nodes for computation from blocks.
                h = features[input_nodes].to(device)
                h_dst = h[: block.number_of_dst_nodes()]

                # calculate node representation layer by layer.
                h = layer(block, (h, h_dst))
                if i != len(self.conv_layers) -1:
                    h = F.relu(h)
                    h = F.dropout(h, p=self.dropout)

                y[output_nodes] = h.cpu()
            
            features = y
            data.barrier()
        
        return y