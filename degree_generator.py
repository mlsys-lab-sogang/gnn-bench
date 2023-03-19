import numpy as np
import pandas as pd

from tqdm import tqdm
from torch_geometric.datasets import Reddit
from ogb.nodeproppred import PygNodePropPredDataset


def generate_degree():
  print("----------------------------")
  print("|     Summary of Reddit    |")
  print("----------------------------")

  degree_reddit = pd.DataFrame()
  dataset = Reddit(root="dataset/reddit")

  # there is only one graph in Node Property Prediction datasets
  print("Length: {}".format(len(dataset)))                          # Length: 1

  data = dataset[0]
  print("Graph: {}".format(data))                                   # Graph: Data(x=[232965, 602], edge_index=[2, 114615892], y=[232965], train_mask=[232965], val_mask=[232965], test_mask=[232965])
  print("Edges: {}".format(data.edge_index))                        # Edges: tensor([[     0,      0,      0,  ..., 232964, 232964, 232964],
                                                                    #                [   242,    249,    524,  ..., 231806, 232594, 232634]])
  print("Number of nodes: {}".format(data.num_nodes))               # Number of nodes: 232965
  print("Number of edges: {}".format(data.num_edges))               # Number of edges: 114615892
  print("Has isolated nodes: {}".format(data.has_isolated_nodes())) # Has isolated nodes: False
  print("Has self loops: {}".format(data.has_self_loops()))         # Has self loops: False
  print("Is directed: {}".format(data.is_directed()))               # Is directed: False

  degrees = np.zeros(shape=(data.num_nodes), dtype=np.int64)

  for idx in tqdm(range(data.num_edges)):
    degrees[data.edge_index[0][idx]] += 1
    degrees[data.edge_index[1][idx]] += 1

  print("degrees.min(): {}".format(degrees.min()))                  # degrees.min(): 2
  print("degrees.max(): {}".format(degrees.max()))                  # degrees.max(): 43314
  print("degrees.std(): {}".format(degrees.std()))                  # degrees.std(): 1599.636410194922

  degree_reddit["node"] = pd.Series(range(data.num_nodes))
  degree_reddit["degree"] = pd.Series(degrees)
  degree_reddit.to_csv("logs/degree_reddit.csv", index=False)

  print("----------------------------")
  print("| Summary of Ogbn-Products |")
  print("----------------------------")

  degree_ogbn_products = pd.DataFrame()
  dataset = PygNodePropPredDataset(name="ogbn-products")

  print("Length: {}".format(len(dataset)))                          # Length: 1

  data = dataset[0]
  print("Graph: {}".format(data))                                   # Graph: Data(num_nodes=2449029, edge_index=[2, 123718280], x=[2449029, 100], y=[2449029, 1])
  print("Edges: {}".format(data.edge_index))                        # Edges: tensor([[      0,  152857,       0,  ..., 2449028,   53324, 2449028],
                                                                    #                [ 152857,       0,   32104,  ...,  162836, 2449028,   53324]])
  print("Number of nodes: {}".format(data.num_nodes))               # Number of nodes: 2449029
  print("Number of edges: {}".format(data.num_edges))               # Number of edges: 123718280
  print("Has isolated nodes: {}".format(data.has_isolated_nodes())) # Has isolated nodes: True
  print("Has self loops: {}".format(data.has_self_loops()))         # Has self loops: True
  print("Is directed: {}".format(data.is_directed()))               # Is directed: False

  degrees = np.zeros(shape=(data.num_nodes), dtype=np.int64)

  for idx in tqdm(range(data.num_edges)):
    degrees[data.edge_index[0][idx]] += 1
    degrees[data.edge_index[1][idx]] += 1

  print("degrees.min(): {}".format(degrees.min()))                  # degrees.min(): 0
  print("degrees.max(): {}".format(degrees.max()))                  # degrees.max(): 34962
  print("degrees.std(): {}".format(degrees.std()))                  # degrees.std(): 191.80998979272695

  degree_ogbn_products["node"] = pd.Series(range(data.num_nodes))
  degree_ogbn_products["degree"] = pd.Series(degrees)
  degree_ogbn_products.to_csv("logs/degree_ogbn_products.csv", index=False)

  print("----------------------------")
  print("|  Summary of Ogbn-Papers  |")
  print("----------------------------")

  degree_ogbn_papers100M = pd.DataFrame()
  dataset = PygNodePropPredDataset(name="ogbn-papers100M")

  print("Length: {}".format(len(dataset)))

  data = dataset[0]
  print("Graph: {}".format(data))
  print("Edges: {}".format(data.edge_index))
  print("Number of nodes: {}".format(data.num_nodes))
  print("Number of edges: {}".format(data.num_edges))
  print("Has isolated nodes: {}".format(data.has_isolated_nodes()))
  print("Has self loops: {}".format(data.has_self_loops()))
  print("Is directed: {}".format(data.is_directed()))

  degrees = np.zeros(shape=(data.num_nodes), dtype=np.int64)

  for idx in tqdm(range(data.num_edges)):
    degrees[data.edge_index[0][idx]] += 1
    degrees[data.edge_index[1][idx]] += 1

  print("degrees.min(): {}".format(degrees.min()))
  print("degrees.max(): {}".format(degrees.max()))
  print("degrees.std(): {}".format(degrees.std()))

  degree_ogbn_papers100M["node"] = pd.Series(range(data.num_nodes))
  degree_ogbn_papers100M["degree"] = degrees
  degree_ogbn_papers100M.to_csv("logs/degree_ogbn_papers100M.csv", index=False)


if __name__ == "__main__":
  generate_degree()
