#!/usr/bin/env python
# coding: utf-8

# In[5]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import liana as li
import anndata
import scanpy as sc
from openproblems import tasks
import os 


# In[6]:


# Install required packages.
import os
import torch
os.environ['TORCH'] = torch.__version__
print(torch.__version__)

# !pip install -q torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html
# !pip install -q torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html
# !pip install -q git+https://github.com/pyg-team/pytorch_geometric.git

# Helper function for visualization.
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from utils import *


# In[7]:


import argparse
import os.path as osp

import torch
import torch.nn.functional as F

from torch_geometric.datasets import Entities
from torch_geometric.nn import FastRGCNConv, RGCNConv, GCNConv, InnerProductDecoder, GAE, VGAE
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


# # Pre-processing from original matrix
# 

# In[8]:


adata = sc.read_h5ad("/h/soemily/GAT/data/Sikander_data/Visium-FZ_GT_P19.h5ad")


# In[9]:


matrix = pd.DataFrame.sparse.from_spmatrix(adata.X,index=adata.obs.index.tolist(),columns=adata.var["feature_name"].tolist())

meta = pd.DataFrame({"cell":adata.obs.index.tolist(),"labels":adata.obs["cell_type_original"].tolist()})
meta.index = meta["cell"].tolist()


# In[11]:


matrix = matrix.transpose()


# In[12]:


matrix = matrix[meta.index.tolist()]


# In[13]:


from scipy.stats import wilcoxon
import anndata
import scanpy as sc


# In[14]:


index = matrix.index.tolist()


# In[15]:
matrix,meta,nodes,interactions,LR_nodes,Omnipath_network = make_nodes_interactions(matrix,input_meta = meta)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"


# LR_nodes = pd.read_csv("/h/soemily/GAT/data/LR_database/OmniPath_nodes.csv",index_col=0)
# Omnipath_network = pd.read_csv("/h/soemily/GAT/data/LR_database/OmniPath_interactions.csv",index_col=0)

# new_identifier = [row["identifier"] + "_" + row["category"] for index,row in LR_nodes.iterrows()]


# LR_nodes["identifier"] = new_identifier
# Omnipath_network["Src"] = LR_nodes.loc[Omnipath_network["Src"].tolist()]["identifier"].tolist()
# Omnipath_network["Dst"] = LR_nodes.loc[Omnipath_network["Dst"].tolist()]["identifier"].tolist()
# LR_nodes = LR_nodes[(LR_nodes["identifier"].isin(ligand_list)) | (LR_nodes["identifier"].isin(receptor_list))]

# Omnipath_network = Omnipath_network[(Omnipath_network["Src"].isin(LR_nodes["identifier"].tolist())) & (Omnipath_network["Dst"].isin(LR_nodes["identifier"].tolist()))]


# LR_nodes["Id"] = range(0,LR_nodes.shape[0])

# LR_nodes.index = LR_nodes["identifier"].tolist()
# Omnipath_network["Src"] = LR_nodes.loc[Omnipath_network["Src"].tolist()]["Id"].tolist()
# Omnipath_network["Dst"] = LR_nodes.loc[Omnipath_network["Dst"].tolist()]["Id"].tolist()


# LR_nodes = LR_nodes[(LR_nodes["Id"].isin(Omnipath_network["Src"].tolist())) | (LR_nodes["Id"].isin(Omnipath_network["Dst"].tolist()))]

# LR_nodes.index = LR_nodes["Id"].tolist()
# Omnipath_network["Src"] = LR_nodes.loc[Omnipath_network["Src"].tolist()]["identifier"].tolist()
# Omnipath_network["Dst"] = LR_nodes.loc[Omnipath_network["Dst"].tolist()]["identifier"].tolist()



# LR_nodes.index = LR_nodes["identifier"].tolist()
# LR_nodes["Id"] = range(0,LR_nodes.shape[0])
# Omnipath_network["Src"] = LR_nodes.loc[Omnipath_network["Src"].tolist()]["Id"].tolist()
# Omnipath_network["Dst"] = LR_nodes.loc[Omnipath_network["Dst"].tolist()]["Id"].tolist()


# In[64]:


from torch_geometric.nn import Node2Vec
import os.path as osp
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.datasets import Planetoid
from tqdm.notebook import tqdm

if (Omnipath_network["Src"].dtype != 'float64') and (Omnipath_network["Src"].dtype != 'int64'):
    LR_nodes.index = LR_nodes["identifier"].tolist()
    LR_nodes["Id"] = range(0,LR_nodes.shape[0])
    Omnipath_network["Src"] = LR_nodes.loc[Omnipath_network["Src"].tolist()]["Id"].tolist()
    Omnipath_network["Dst"] = LR_nodes.loc[Omnipath_network["Dst"].tolist()]["Id"].tolist()
Omnipath_data,Omnipath_nodes,Omnipath_interactions = make_dataset(LR_nodes,Omnipath_network,first=False,pathway_encode=False)
Omnipath_nodes.index = Omnipath_nodes["Id"].tolist()


df_list = []


df_list = []

for i in range(10):
    print(i)
    Omnipath_network["Src"] = Omnipath_network["Src"].sample(frac=1).tolist()
    Omnipath_network["Dst"] = Omnipath_network["Dst"].sample(frac=1).tolist()
    if (Omnipath_network["Src"].dtype != 'float64') and (Omnipath_network["Src"].dtype != 'int64'):
        LR_nodes.index = LR_nodes["identifier"].tolist()
        LR_nodes["Id"] = range(0,LR_nodes.shape[0])
        Omnipath_network["Src"] = LR_nodes.loc[Omnipath_network["Src"].tolist()]["Id"].tolist()
        Omnipath_network["Dst"] = LR_nodes.loc[Omnipath_network["Dst"].tolist()]["Id"].tolist()
    print("Done Preprocessing")
    df = get_Omnipath_embeddings(LR_nodes,Omnipath_network)
    print("Done Omnipath embeddings")
    interactions["Src"] = interactions["Src"].sample(frac=1).tolist()
    interactions["Dst"] = interactions["Dst"].sample(frac=1).tolist()  
    Omnipath_nodes.index = Omnipath_nodes["identifier"].tolist()
    df_list.append(get_cell_LR_embeddings(matrix,meta,nodes,interactions,df,Omnipath_nodes,Omnipath_interactions,spatial="/h/soemily/GAT/data/Sikander_data/Visium-FZ_GT_P19.h5ad"))
    print("Done cell embeddings")


# In[74]:


for i in range(len(df_list)):
    temp = df_list[i]
    temp =temp.drop_duplicates(["Src Cell","Dst Cell"])
    temp = temp[temp["Src Cell"] != temp["Dst Cell"]]
    temp = temp.head(1000)
    temp.to_csv(f"/h/soemily/GAT/results/Sikander_data/spatial_{i+91}_random.csv")

