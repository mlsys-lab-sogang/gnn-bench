import sys

import numpy as np
import pandas as pd

from tqdm import tqdm
from torch_geometric.datasets import Reddit
from ogb.nodeproppred import PygNodePropPredDataset


d_names = ["reddit", "ogbn-products", "ogbn-papers100M"]

def generate_degree(d_name):
  dataset = Reddit(root="dataset/reddit") if d_name == d_names[0] else PygNodePropPredDataset(name=d_name)

  # there is only one graph in Node Property Prediction datasets
  print(f"Length: {len(dataset)}") # Length: 1

  graph = dataset[0]
  print(f"Graph: {graph}")                                   # Graph: Data(num_nodes=2449029, edge_index=[2, 123718280], x=[2449029, 100], y=[2449029, 1])
  print(f"Edges: {graph.edge_index}")                        # Edges: tensor([[      0,  152857,       0,  ..., 2449028,   53324, 2449028],
                                                             #                [ 152857,       0,   32104,  ...,  162836, 2449028,   53324]])
  print(f"Number of nodes: {graph.num_nodes}")               # Number of nodes: 2449029
  print(f"Number of edges: {graph.num_edges}")               # Number of edges: 123718280
  print(f"Has isolated nodes: {graph.has_isolated_nodes()}") # Has isolated nodes: True
  print(f"Has self loops: {graph.has_self_loops()}")         # Has self loops: True
  print(f"Is directed: {graph.is_directed()}")               # Is directed: False

  degrees = np.zeros(shape=(graph.num_nodes), dtype=np.int64)

  for idx in tqdm(range(graph.num_edges)):
    degrees[graph.edge_index[0][idx]] += 1
    degrees[graph.edge_index[1][idx]] += 1

  print(f"degrees.min(): {degrees.min()}") # degrees.min(): 0
  print(f"degrees.max(): {degrees.max()}") # degrees.max(): 34962
  print(f"degrees.std(): {degrees.std()}") # degrees.std(): 191.80998979272695

  pd.DataFrame({"index": range(graph.num_nodes), "degree": degrees}).to_csv("logs/degree_{}.csv".format(d_name.replace("-", "_")), index=False)

def main(argv):
  if len(argv) != 2:
    sys.exit("Expected one argument but got {}.".format(len(argv)-1))
  if argv[1] not in d_names:
    sys.exit("Dataset should be one of:\n\t{}".format('\t'.join(d_names)))
  generate_degree(argv[1])

if __name__ == "__main__":
  main(sys.argv)
