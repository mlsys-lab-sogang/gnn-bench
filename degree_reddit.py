import numpy as np
import pandas as pd

from tqdm import tqdm
from torch_geometric.datasets import Reddit


dataset = Reddit(root="dataset/reddit")

print(f"Number of nodes: {dataset.data.num_nodes}")                    # Number of nodes: 232965
print(f"Number of edges: {dataset.data.num_edges}")                    # Number of edges: 114615892
print(f"Contains isolated nodes: {dataset.data.has_isolated_nodes()}") # Contains isolated nodes: False
print(f"Contains self loops: {dataset.data.has_self_loops()}")         # Contains self loops: False
print(f"Is directed: {dataset.data.is_directed()}")                    # Is directed: False
print(f"Length: {len(dataset)}")                                       # Length: 1
print(dataset[0])                                                      # Data(x=[232965, 602], edge_index=[2, 114615892], y=[232965], train_mask=[232965], val_mask=[232965], test_mask=[232965])
print(dataset.data.edge_index)                                         # tensor([[     0,      0,      0,  ..., 232964, 232964, 232964],
                                                                       #         [   242,    249,    524,  ..., 231806, 232594, 232634]])
degrees = np.zeros(shape=(dataset.data.num_nodes), dtype=np.int64)

for idx in tqdm(range(dataset.data.num_edges)):
  degrees[dataset.data.edge_index[0][idx]] += 1
  degrees[dataset.data.edge_index[1][idx]] += 1

print(f"degrees.min(): {degrees.min()}") # degrees.min(): 2
print(f"degrees.max(): {degrees.max()}") # degrees.max(): 43314
print(f"degrees.std(): {degrees.std()}") # degrees.std(): 1599.636410194922

pd.DataFrame({"index": range(dataset.data.num_nodes), "degree": degrees}).to_csv("logs/degree_reddit.csv", index=False)
