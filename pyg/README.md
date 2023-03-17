# Model & Scripts built with PyTorch Geometric (PyG)

This repository contains some benchmarking codes with PyG, using `Reddit`, `ogbn-products` dataset.

Original source codes are from [PyTorch Geometric official examples](https://github.com/pyg-team/pytorch_geometric/tree/master/examples) and [OGB Nodeproppred examples](https://github.com/snap-stanford/ogb/tree/master/examples/nodeproppred/products).

---------

## Models 

Mainly, we are using **`GraphSAGE` model**, proposed from [this paper](https://arxiv.org/abs/1706.02216).

We are mainly using PyG's **`NeigborLoader` class** to perform mini-batch training via node sampling.

Additionally we applied other two additional model, `GAT` model from [this paper](https://arxiv.org/abs/1710.10903) and `ClusterGCN` model from [this paper](https://arxiv.org/abs/1905.07953) to check mini-batch training.

> `GAT` mini-batch training uses node sampling, and `ClusterGCN` mini-batch training uses subgraph sampling (`ClusterData`, `ClusterLoader`).

---------

## Code examples

We provides some test benchmarking codes.

All codes (except `reddit_clustergcn.py`) uses PyG's `NeighborLoader` class for mini-batch training.

### Single-GPU examples

 - **[ogbn_products_sage_edit.py](https://github.com/mlsys-lab-sogang/gnn-bench/blob/master/pyg/ogbn_products_sage_edit.py)** : Performs **full-batch train** & **mini-batch train** to `ogbn-products` dataset, with using `GraphSAGE` model.
 - **[reddit_sage.py](https://github.com/mlsys-lab-sogang/gnn-bench/blob/master/pyg/reddit_sage.py)** : Performs **full-batch train** & **mini-batch train** to `Reddit` dataset, with using `GraphSAGE` model.
 - **[reddit_gat.py](https://github.com/mlsys-lab-sogang/gnn-bench/blob/master/pyg/reddit_gat.py)** : Performs **mini-batch train** to `Reddit` dataset, with using `GAT` model.
 - **[reddit_clustergcn.py](https://github.com/mlsys-lab-sogang/gnn-bench/blob/master/pyg/reddit_clustergcn.py)** : Performs **mini-batch train** to `Reddit` dataset, with using `ClusterGCN` model.

> `ogbn_products_sage_edit.py` and `reddit_sage.py` will make some result .csv files to specific directory (**`'gnn-bench/logs/single_gpu/'`**)

<details>
<summary> <b><u><font size='+2'> Script running examples </font></u></b> </summary>
<div markdown='1'>

For all Single-GPU examples, we commonly use following arguments:

 - `device` : which GPU device to use. (`default=0`)
 - `num_layers` : defines how many to stack layers. (`default=3`)
 - `hidden_channels` : # of hidden units in hidden layers. (`default=128`)
 - `epochs` : # of train/test iterations. (`default=300`)
 - `dropout` : dropout rate.
    <details>
    <summary> default dropout rate of each scripts </summary>
    <div markdown='1'>

    -  `default=0.3` for `reddit_sage.py`
    -  `default=0.5` for `ogbn_products_sage_edit.py`
    -  `default=0.6` for `reddit_gat.py`
    -  `default=0.2` for `reddit_clustergcn.py`

    </div>
    </details>
 - `lr` : learning rate. (`default=0.01` // `default=0.005` for `reddit_gat.py`)
 - `train_type` : which train type to run script. **only available in `reddit_sage.py` and `ogbn_products_sage_edit.py`**.
   - must specify `full` for full-batch training or `mini` for mini-batch training
 - `fanout` : # of neighbors to sample for each node in each iteration.
 - `batch_size` : # of anchor nodes in each mini-batch.
   - Each anchor nodes will make computation graph by sampling neighbors, following `fanout`.
   - For `reddit_clustergcn.py`, this means # of partitioned subgraphs in each mini-batch. (`default=20`)
  
For some examples, we use following additional arguments:

 - `heads` in `reddit_sage.py` : # of multi-head attention heads. (`default=2`)
 - `num_parts` in `reddit_clustergcn.py` : # of partitions of entire graph. (`default=1500`)

<br>

For `reddit_sage.py` and `ogbn_products_sage_edit.py`, form is like below: 

```
python [FILE_NAME] --device [DEVICE] --num_layers [NUM_LAYERS] --hidden_channels [HIDDEN_CHANNELS] --dropout [DROPOUT] --lr [LR] --epochs [EPOCHS] --train_type [TRAIN_TYPE] --fanout [FANOUT] --batch_size [BATCH_SIZE]
```
<br>

For example, if using `reddit_sage.py` with **mini-batch manner**, we can write script like below : 

```
python reddit_sage.py --num_layers 3 --hidden_channels 128 --dropout 0.3 --lr 0.01 -- epochs 300 --train_type mini --fanout 15 10 5 --batch_size 1024
```
 - In this case, we stack **GraphSAGE model with 3 layers**, set hidden layer's unit size to 128.
 - And we set **mini-batch** train by `--train_type mini`, set **number of neighbors to sample for each node in each iteration** to `[15, 10, 5]` by `--fanout 15 10 5`. 

<br>

If using `ogbn_products_sage_edit.py` with **mini-batch manner**, we can write script as same as above's `reddit_sage.py` case like below:

```
python ogbn_products_sage_edit.py --num_layers 3 --hidden_channels 128 --dropout 0.3 --lr 0.01 -- epochs 300 --train_type mini --fanout 15 10 5 --batch_size 1024
```

<br>

If using `reddit_sage.py` or `ogbn_products_sage_edit.py` with **full-batch manner**, we can write script as follow:

```
python FILE_NAME --num_layers 3 --hidden_channels 128 --dropout 0.3 --lr 0.01 --epochs 300 --train_type full
```

<br>

If using `reddit_gat.py` or `reddit_clustergcn.py` we can write each script like below:

```
python reddit_gat.py --num_layers 3 --hidden_channels 128 --dropout 0.6 --lr 0.005 --epochs 300 --fanout 15 10 5 --batch_size 1024 --heads 2
```
```
python reddit_clustergcn.py --num_layers 3 --hidden_channels 128 --dropout 0.2 --lr 0.01 --epochs 300 --num_parts 1500 --batch_size 20
```


</div>
</details>

<br>

### Multi-GPU examples

> We use PyTorch's **`DistributedDataParallel`** for **single machine - multi GPU** training.

 - **[ogbn_products_sage_dist.py](https://github.com/mlsys-lab-sogang/gnn-bench/blob/master/pyg/ogbn_products_sage_dist.py)** : Performs **distributed mini-batch train** to `ogbn-products` dataset, with using `GraphSAGE` model.
 - **[reddit_sage_dist.py](https://github.com/mlsys-lab-sogang/gnn-bench/blob/master/pyg/reddit_sage_dist.py)** : Performs **distributed mini-batch train** to `Reddit` dataset, with using `GraphSAGE` model.

> Both files will make some result .csv files to specific directory (**`'gnn-bench/logs/multi_gpu/'`**), and **each GPU will make in-batch result files**.  

<details>
<summary> <b><u><font size='+2'> Script running examples </font></u></b> </summary>
<div markdown='1'>

For all Multi-GPU examples, we commonly use following arguments:

 - `num_layers` : defines how many to stack layers. (`default=3`)
 - `hidden_channels` : # of hidden units in hidden layers. (`default=128`)
 - `epochs` : # of train/test iterations. (`default=300`)
 - `dropout` : dropout rate. (`default=0.3`)
 - `lr` : learning rate. (`default=0.01`)
 - `fanout` : # of neighbors to sample for each node in each iteration.
 - `batch_size` : # of anchor nodes in each mini-batch.
   - Each anchor nodes will make computation graph by sampling neighbors, following `fanout`.

For `ogbn_products_sage_dist.py`, like OGB's official example, we leave those arguments:

 - `log_steps` : how often to print training epoch's result. (`default=1`)
 - `runs` : # of independent experiments. (`default=1`)
  

<br>

For both script's form is like below: 

```
python [FILE_NAME] --num_layers [NUM_LAYERS] --hidden_channels [HIDDEN_CHANNELS] --dropout [DROPOUT] --lr [LR] --epochs [EPOCHS] --fanout [FANOUT] --batch_size [BATCH_SIZE]
```
<br>

For example, if using `reddit_sage_dist.py` or `ogbn_products_sage_dist.py`, we can write script like below : 

```
python reddit_sage_dist.py --num_layers 3 --hidden_channels 128 --dropout 0.3 --lr 0.01 -- epochs 100 --fanout 15 10 5 --batch_size 1024
```
```
python ogbn_products_sage_dist.py --num_layers 3 --hidden_channels 128 --dropout 0.3 --lr 0.01 -- epochs 100 --fanout 15 10 5 --batch_size 1024
```

 - In both case, we stack **GraphSAGE model with 3 layers**, set hidden layer's unit size to 128.
 - And we set **number of neighbors to sample for each node in each iteration** to `[15, 10, 5]` by `--fanout 15 10 5`. 


<br>

> Also we can run those scripts by **torchrun** style.
- We provide some **torchrun** shell scripts, so we can use it.  
- And each .sh files uses different `fanout`.   
- Just type like this : `./reddit_dist_fanout_15_10_5.sh NUM_GPUS_IN_LOCAL`    
- For example, if we have 4 GPU in single machine, command will like this : `./reddit_dist_fanout_15_10_5.sh 4`


</div>
</details>

---------

## References

- [PyTorch Geometric official examples](https://github.com/pyg-team/pytorch_geometric/tree/master/examples) 
- [OGB Nodeproppred examples](https://github.com/snap-stanford/ogb/tree/master/examples/nodeproppred/products)