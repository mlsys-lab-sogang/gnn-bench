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

