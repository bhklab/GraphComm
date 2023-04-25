





import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import liana as li
import anndata
import scanpy as sc
import os 
from utils import *
from model import *
import argparse
import os.path as osp

import torch
import torch.nn.functional as F

from torch_geometric.datasets import Entities
from torch_geometric.nn import FastRGCNConv, RGCNConv, GCNConv, GAE, VGAE
from torch_geometric.utils import k_hop_subgraph

from torch_geometric.datasets import Planetoid
from torch_geometric.data.data import Data
from torch_geometric.transforms import NormalizeFeatures


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import SpectralClustering
import random
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity

from scipy.io import mmread

import argparse

    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"
    
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='{args.dataset}', help='Dataset to run GraphComm on')
parser.add_argument('--spatial', type=str, default=None, help="pathway to h5ad containing spatial coordinates")
parser.add_argument("--reproduce",type=str,default="True",help="reproduce original results of paper")
parser.add_argument("--Omnipath_lr",type=str,default="0.01",help="reproduce original results of paper")

args = parser.parse_args()

nodes = pd.read_csv(f"/data/GraphComm_Input/{args.dataset}/nodes.csv",index_col=0)
interactions = pd.read_csv(f"/data/GraphComm_Input/{args.dataset}/interactions.csv",index_col=0)
matrix = pd.read_csv(f"/data/GraphComm_Input/{args.dataset}/matrix.csv",index_col=0)
meta = pd.read_csv(f"/data/GraphComm_Input/{args.dataset}/meta.csv",index_col=0)
LR_nodes = pd.read_csv(f"/data/GraphComm_Input/{args.dataset}/LR_nodes.csv",index_col=0)
Omnipath_network = pd.read_csv(f"/data/GraphComm_Input/{args.dataset}/Omnipath_network.csv",index_col=0)

Omnipath_data,Omnipath_nodes,Omnipath_interactions = make_dataset(LR_nodes,Omnipath_network,first=False,pathway_encode=False)
print("Getting embeddings from ground truth")
if args.reproduce == "True":
    total_embeddings_df = get_Omnipath_embeddings(LR_nodes,Omnipath_network,reproduce=args.dataset,lr=float(args.Omnipath_lr))
else:
    total_embeddings_df = get_Omnipath_embeddings(LR_nodes,Omnipath_network,reproduce=None,save=args.dataset,lr=float(args.Omnipath_lr))
print("getting GAT cell communication probabilities")
if args.reproduce == "True":
    total_link_df = get_cell_LR_embeddings(matrix,meta,nodes,interactions,total_embeddings_df,Omnipath_nodes,Omnipath_interactions,spatial=args.spatial,reproduce=args.dataset)

else:
    total_link_df = get_cell_LR_embeddings(matrix,meta,nodes,interactions,total_embeddings_df,Omnipath_nodes,Omnipath_interactions,spatial=args.spatial,reproduce=None,save=args.dataset)


os.system("mkdir -p {}".format(f"/results/GraphComm_Output/{args.dataset}/"))
total_link_df.to_csv(f"/results/GraphComm_Output/{args.dataset}/CCI.csv")

