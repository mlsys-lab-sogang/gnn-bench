r"""
Model checkpointing script for GraphSAGE with Reddit.
"""
import argparse

import torch

from torch_geometric.datasets import Reddit
from reddit_sage_dist import SAGE_Dist


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--num_layers', type=int, default=3)
  parser.add_argument('--hidden_channels', type=int, default=256)
  parser.add_argument('--dropout', type=float, default=0.3)
  args = parser.parse_args()

  dataset = Reddit(root="../dataset/reddit/")

  model = SAGE_Dist(in_channels=dataset.num_features, # 602
                    hidden_channels=args.hidden_channels,
                    out_channels=dataset.num_classes, # 41
                    num_layers=args.num_layers,
                    dropout=args.dropout)

  torch.save(model.state_dict(), f="../checkpoints/reddit_sage.pt")
