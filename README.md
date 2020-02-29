# Inductive Representation Learning on Temporal Graphs

## Introduction

The evolving nature of temporal dynamic graphs requires handling new nodes as well as capturing temporal patterns. The node embeddings, as functions of time, should represent both the static node features and the evolving topological structures. 

We propose the temporal graph attention (TGAT) layer to efficiently aggregate temporal-topological neighborhood features as well as to learn the time-feature interactions. Stacking TGAT layers, the network recognizes the node embeddings as functions of time and is able to inductively infer embeddings for both new and observed nodes as the graph evolves. 

The proposed approach handles both node classification and link prediction task, and can be naturally extended to include the temporal edge features.


#### Paper link: [Inductive Representation Learning on Temporal Graphs](https://openreview.net/attachment?id=rJeW1yHYwH&name=original_pdf)


## Running the experiments

### Dataset and preprocessing

#### Download the public data
* [Reddit](http://snap.stanford.edu/jodie/reddit.csv)

* [Wikipedia](http://snap.stanford.edu/jodie/wikipedia.csv)

#### Preprocess the data
We use the dense `npy` format to save the features in binary format. If edge features or nodes features are absent, it will be replaced by a vector of zeros. 
```{bash}
python process.py 
```

#### Use your own data
Put your data under `processed` folder. The required input data includes `ml_${DATA_NAME}.csv`, `ml_${DATA_NAME}.npy` and `ml_${DATA_NAME}_node.npy`. They store the edge linkages, edge features and node features respectively. 

The `CSV` file has following columns
```
u, i, ts, label, idx
```
, which represents source node index, target node index, time stamp, edge label and the edge index. 

`ml_${DATA_NAME}.npy` has shape of [#temporal edges + 1, edge features dimention]. Similarly, `ml_${DATA_NAME}_node.npy` has shape of [#nodes + 1, node features dimension].


All node index starts from `1`. The zero index is reserved for `null` during padding operations. So the maximum of node index equals to the total number of nodes. Similarly, maxinum of edge index equals to the total number of temporal edges. The padding embeddings or the null embeddings is a vector of zeros.

### Requirements

* python >= 3.7

* Dependency

```{bash}
pandas==0.24.2
torch==1.1.0
tqdm==4.41.1
numpy==1.16.4
scikit_learn==0.22.1
```

### Command and configurations

#### Sample commend

* Learning the network using link prediction tasks
```{bash}
# t-gat learning on wikipedia data
python -u learn_edge.py -d wikipedia --bs 200 --uniform  --n_degree 20 --agg_method attn --attn_mode prod --gpu 0 --n_head 2 --prefix hello_world

# t-gat learning on reddit data
python -u learn_edge.py -d reddit --bs 200 --uniform  --n_degree 20 --agg_method attn --attn_mode prod --gpu 0 --n_head 2 --prefix hello_world
```

* Learning the down-stream task (node-classification)

Node-classification task reuses the network trained previously. Make sure the `prefix` is the same so that the checkpoint can be found under `saved_models`.

```{bash}
# on wikipedia
python -u learn_node.py -d wikipedia --bs 100 --uniform  --n_degree 20 --agg_method attn --attn_mode prod --gpu 0 --n_head 2 --prefix hello_world

# on reddit
python -u learn_node.py -d reddit --bs 100 --uniform  --n_degree 20 --agg_method attn --attn_mode prod --gpu 0 --n_head 2 --prefix hello_world
```
#### General flags

```{txt}
optional arguments:
  -h, --help            show this help message and exit
  -d DATA, --data DATA  data sources to use, try wikipedia or reddit
  --bs BS               batch_size
  --prefix PREFIX       prefix to name the checkpoints
  --n_degree N_DEGREE   number of neighbors to sample
  --n_head N_HEAD       number of heads used in attention layer
  --n_epoch N_EPOCH     number of epochs
  --n_layer N_LAYER     number of network layers
  --lr LR               learning rate
  --drop_out DROP_OUT   dropout probability
  --gpu GPU             idx for the gpu to use
  --node_dim NODE_DIM   Dimentions of the node embedding
  --time_dim TIME_DIM   Dimentions of the time embedding
  --agg_method {attn,lstm,mean}
                        local aggregation method
  --attn_mode {prod,map}
                        use dot product attention or mapping based
  --time {time,pos,empty}
                        how to use time information
  --uniform             take uniform sampling from temporal neighbors
```

## Cite us

```
@inproceedings{tgat_iclr20,
title={Inductive representation learning on temporal graphs},
author={da Xu and chuanwei ruan and evren korpeoglu and sushant kumar and kannan achan},
booktitle={International Conference on Learning Representations (ICLR)},
year={2020}
}
```


