#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import liana as li
import anndata
import scanpy as sc
from openproblems import tasks
import os 


# In[2]:


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


# In[3]:


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

from scipy.stats import wilcoxon
import anndata
import scanpy as sc


# In[48]:




# In[49]:


from torch_geometric.nn import GATConv
class GAT(torch.nn.Module):
    def __init__(self,data,num_classes=3):
        super(GAT, self).__init__()
        self.hid = 3
        self.in_head = 3
        self.out_head = 3
        
        
        self.conv1 = GATConv(data.x.shape[1], self.hid, heads=self.in_head, dropout=0.6)
        self.conv2 = GATConv(self.hid*self.in_head,num_classes, concat=False,
                             heads=self.out_head, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
                
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.log_softmax(x,dim=1)
        return x
    
    
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"
    


# # Get Omnipath embedding

# In[54]:




# In[65]:


from torch_geometric.nn import Node2Vec
import os.path as osp
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.datasets import Planetoid
from tqdm.notebook import tqdm


# In[66]:


def train(model,loader,optimizer):
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in (loader):
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# In[67]:


@torch.no_grad()
def test():
    model.eval()
    z = model()
    acc = model.test(z[data.train_mask], data.y[data.train_mask],
                     z[data.test_mask], data.y[data.test_mask],
                     max_iter=150)
    return acc


# In[68]:


# In[50]:


def get_Omnipath_embeddings(nodes,interactions):
    Omnipath_data,Omnipath_nodes,Omnipath_interactions = make_dataset(nodes,interactions,first=False,pathway_encode=False)
    node_info = pd.DataFrame(np.zeros((Omnipath_nodes.shape[0],Omnipath_nodes.shape[0])),index=Omnipath_nodes["identifier"].tolist(),columns=Omnipath_nodes["identifier"].tolist())

    temp_identifiers = [i.split("_")[0] for i in Omnipath_nodes["identifier"].tolist()]

    complexes = pd.read_csv("/data/LR_database/complexes.csv")
    complexes = complexes[complexes["member"].isin(temp_identifiers)]

    temp_nodes = Omnipath_nodes.copy()
    temp_nodes.index = temp_identifiers
    temp_nodes = temp_nodes[~temp_nodes.index.duplicated(keep='first')]

    complexes["member"] = temp_nodes.loc[complexes["member"].tolist()]["identifier"].tolist()

    group_complex = complexes.groupby("complex").agg(list)

    group_complex.index=range(0,group_complex.shape[0])

    for index,row in group_complex.iterrows():
        node_info.loc[list(set(row["member"])),list(set(row["member"]))] = index

    # for i in group_complex["member"].tolist():
    #     node_info.loc[list(set(i)),list(set(i))] = 1

    pathways = pd.read_csv("/data/kegg_pathways.csv",index_col=0)
    pathways = pathways[pathways["genesymbol"].isin(temp_identifiers)]
    pathways["genesymbol"] = temp_nodes.loc[pathways["genesymbol"].tolist()]["identifier"].tolist()
    group_pathway = pathways.groupby("pathway").agg(list)

    group_pathway.index=range(0,group_pathway.shape[0])

    for index,row in group_pathway.iterrows():
        node_info.loc[list(set(row["genesymbol"])),list(set(row["genesymbol"]))] += index

    # for i in group_pathway["genesymbol"].tolist():
    #     node_info.loc[list(set(i)),list(set(i))] += 1

    truth_info = pd.DataFrame(np.zeros((Omnipath_nodes.shape[0],Omnipath_nodes.shape[0])),index=Omnipath_nodes["identifier"].tolist(),columns=Omnipath_nodes["identifier"].tolist())

    Omnipath_nodes.index = Omnipath_nodes["Id"].tolist()

    ident_interactions = Omnipath_interactions.copy()
    ident_interactions["Src"] = Omnipath_nodes.loc[ident_interactions["Src"].tolist()]["identifier"].tolist()
    ident_interactions["Dst"] = Omnipath_nodes.loc[ident_interactions["Dst"].tolist()]["identifier"].tolist()

    for index,row in ident_interactions.iterrows():
        truth_info.loc[row["Src"],row["Dst"]] = 1

    ligands = Omnipath_nodes[Omnipath_nodes["category"]=="Ligand"]["identifier"].tolist()
    receptors = Omnipath_nodes[Omnipath_nodes["category"]=="Receptor"]["identifier"].tolist()
    #truth_info = truth_info.loc[ligands,receptors]
    truth_info = torch.Tensor(truth_info.values).to(device)


    ligands = Omnipath_nodes[Omnipath_nodes["category"]=="Ligand"]["Id"].tolist()
    receptors = Omnipath_nodes[Omnipath_nodes["category"]=="Receptor"]["Id"].tolist()


    ident_interactions = ident_interactions.drop_duplicates("Src")
    ident_interactions = ident_interactions.drop_duplicates("Dst")

    ident_interactions.index = range(0,ident_interactions.shape[0])

    truth_list = []
    for i in Omnipath_nodes["identifier"].tolist():
        if "Ligand" in i:
            if i in ident_interactions["Src"].tolist():
                #truth_list.append(ident_interactions[ident_interactions["Src"]==i].index.tolist()[0] + 1)
                truth_list.append(1)
            else:
                truth_list.append(0)
        if "Receptor" in i:
            if i in ident_interactions["Dst"].tolist():
                #truth_list.append(ident_interactions[ident_interactions["Dst"]==i].index.tolist()[0] + 1)
                truth_list.append(1)
            else:
                truth_list.append(0)

    node_info.values[np.where(np.isnan(node_info.values))] = 0
    node_info.values[np.where(np.isinf(node_info.values))] = 0

    Omnipath_data.x = torch.Tensor(node_info.values)

    Omnipath_data.y = torch.Tensor(truth_list).type(torch.LongTensor)
    #Omnipath_data.y = truth_info

    Omnipath_nodes.index = Omnipath_nodes["Id"].tolist()

    Omnipath_interactions["Src"] = [Omnipath_nodes.loc[i]["identifier"] for i in Omnipath_interactions["Src"].tolist()]
    Omnipath_interactions["Dst"] = [Omnipath_nodes.loc[i]["identifier"] for i in Omnipath_interactions["Dst"].tolist()]

    #edge_weights = [max(node_info.loc[i,j],node_info.loc[j,i]) for i,j in zip(Omnipath_interactions["Src"].tolist(),Omnipath_interactions["Dst"].tolist())]
    edge_weights = [1 for i,j in zip(Omnipath_interactions["Src"].tolist(),Omnipath_interactions["Dst"].tolist())]
    data = Omnipath_data
    model = Node2Vec(data.edge_index, embedding_dim=2, walk_length=40,
         context_size=40, walks_per_node=10,
         num_negative_samples=1, p=1, q=1, sparse=True).to(device)

    loader = model.loader(batch_size=2, shuffle=True, num_workers=4)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
    
    
    for epoch in range(100):
        loss = train(model,loader,optimizer)
        #acc = test()
#         if epoch % 10 == 10:
#             print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
    model.eval()
    z = model(torch.arange(data.num_nodes)).detach()

    ligand_ids = Omnipath_nodes[Omnipath_nodes["category"].str.contains("Ligand")]["Id"].tolist()
    receptor_ids = Omnipath_nodes[Omnipath_nodes["category"].str.contains("Receptor")]["Id"].tolist()

    ligand_embeddings = z[ligand_ids,:]
    receptor_embeddings = z[receptor_ids,:]

    total_embeddings = torch.inner(ligand_embeddings,receptor_embeddings)

    total_embeddings_df = pd.DataFrame(total_embeddings.numpy(),index=Omnipath_nodes[Omnipath_nodes["category"].str.contains("Ligand")]["identifier"].tolist(),columns=Omnipath_nodes[Omnipath_nodes["category"].str.contains("Receptor")]["identifier"].tolist())
    return total_embeddings_df


# In[69]:


# In[77]:


df_list = []
# In[91]:

matrix = pd.read_csv("/data/GraphComm_Input/Pre_Post/Day0/matrix.csv",index_col=0)
original_matrix,original_meta,original_nodes,original_interactions,original_LR_nodes,original_Omnipath_network = make_nodes_interactions(matrix)
original_LR_nodes.index = original_LR_nodes["Id"].tolist()
original_Omnipath_network["Src"] = original_LR_nodes.loc[original_Omnipath_network["Src"].tolist()]["identifier"].tolist()
original_Omnipath_network["Dst"] = original_LR_nodes.loc[original_Omnipath_network["Dst"].tolist()]["identifier"].tolist()

matrix = pd.read_csv("/data/GraphComm_Input/Pre_Post/Day7_1/matrix.csv",index_col=0)
first_matrix,first_meta,first_nodes,first_interactions,first_LR_nodes,first_Omnipath_network = make_nodes_interactions(matrix)
first_LR_nodes.index = first_LR_nodes["Id"].tolist()
first_Omnipath_network["Src"] = first_LR_nodes.loc[first_Omnipath_network["Src"].tolist()]["identifier"].tolist()
first_Omnipath_network["Dst"] = first_LR_nodes.loc[first_Omnipath_network["Dst"].tolist()]["identifier"].tolist()

matrix = pd.read_csv("/data/GraphComm_Input/Pre_Post/Day7_2/matrix.csv",index_col=0)
second_matrix,second_meta,second_nodes,second_interactions,second_LR_nodes,second_Omnipath_network = make_nodes_interactions(matrix)
second_LR_nodes.index = second_LR_nodes["Id"].tolist()
second_Omnipath_network["Src"] = second_LR_nodes.loc[second_Omnipath_network["Src"].tolist()]["identifier"].tolist()
second_Omnipath_network["Dst"] = second_LR_nodes.loc[second_Omnipath_network["Dst"].tolist()]["identifier"].tolist()


# In[98]:


rep_1_list = []
rep_2_list = []
day_0_list = []
for i in range(5):
    print(i)
    LR_nodes = pd.concat([first_LR_nodes,second_LR_nodes]).drop_duplicates("identifier")
    LR_nodes["Id"] = range(LR_nodes.shape[0])
    Omnipath_network = pd.concat([first_Omnipath_network,second_Omnipath_network]).drop_duplicates(["Src","Dst"])
    LR_nodes.index = LR_nodes["identifier"].tolist()
    Omnipath_network["Src"] = LR_nodes.loc[Omnipath_network["Src"].tolist()]["Id"].tolist()
    Omnipath_network["Dst"] = LR_nodes.loc[Omnipath_network["Dst"].tolist()]["Id"].tolist()
    Omnipath_network["Src"] = Omnipath_network["Src"].sample(frac=1).tolist()
    Omnipath_network["Dst"] = Omnipath_network["Dst"].sample(frac=1).tolist()
    print("Done Preprocessing")
    df = get_Omnipath_embeddings(LR_nodes,Omnipath_network)
    print("Done Omnipath")
    LR_nodes.index = LR_nodes["identifier"].tolist()
    Omnipath_network["Src"] = LR_nodes.loc[Omnipath_network["Src"].tolist()]["Id"].tolist()
    Omnipath_network["Dst"] = LR_nodes.loc[Omnipath_network["Dst"].tolist()]["Id"].tolist()
    Omnipath_data,Omnipath_nodes,Omnipath_interactions = make_dataset(LR_nodes,Omnipath_network,first=False,pathway_encode=False)
    Omnipath_nodes.index = Omnipath_nodes["identifier"].tolist()
    first_interactions["Src"] = first_interactions["Src"].sample(frac=1).tolist()
    first_interactions["Dst"] = first_interactions["Dst"].sample(frac=1).tolist()  
    second_interactions["Src"] = second_interactions["Src"].sample(frac=1).tolist()
    second_interactions["Dst"] = second_interactions["Dst"].sample(frac=1).tolist()  
    original_interactions["Src"] = original_interactions["Src"].sample(frac=1).tolist()
    original_interactions["Dst"] = original_interactions["Dst"].sample(frac=1).tolist()    
    
    first_genes = list(set(first_matrix.index.tolist())& set(Omnipath_nodes.index.tolist()))
    second_genes = list(set(second_matrix.index.tolist())& set(Omnipath_nodes.index.tolist()))
    original_genes = list(set(original_matrix.index.tolist())& set(Omnipath_nodes.index.tolist()))

    first_Omnipath_nodes = Omnipath_nodes.loc[first_genes]
    second_Omnipath_nodes = Omnipath_nodes.loc[second_genes]
    original_Omnipath_nodes = Omnipath_nodes.loc[original_genes]
    
    first_Omnipath_nodes["Id"] = range(first_Omnipath_nodes.shape[0])
    second_Omnipath_nodes["Id"] = range(second_Omnipath_nodes.shape[0])
    original_Omnipath_nodes["Id"] = range(original_Omnipath_nodes.shape[0])
    
    first_Omnipath_nodes.index = first_Omnipath_nodes["identifier"].tolist()
    second_Omnipath_nodes.index = second_Omnipath_nodes["identifier"].tolist()
    original_Omnipath_nodes.index = original_Omnipath_nodes["identifier"].tolist()
    
    first_Omnipath_interactions = Omnipath_interactions[(Omnipath_interactions["Src"].isin(first_Omnipath_nodes["identifier"].tolist())) & (Omnipath_interactions["Dst"].isin(first_Omnipath_nodes["identifier"].tolist()))]
    second_Omnipath_interactions = Omnipath_interactions[(Omnipath_interactions["Src"].isin(second_Omnipath_nodes["identifier"].tolist())) & (Omnipath_interactions["Dst"].isin(second_Omnipath_nodes["identifier"].tolist()))]
    original_Omnipath_interactions = Omnipath_interactions[(Omnipath_interactions["Src"].isin(original_Omnipath_nodes["identifier"].tolist())) & (Omnipath_interactions["Dst"].isin(original_Omnipath_nodes["identifier"].tolist()))]
    
    day_0_list.append(get_cell_LR_embeddings(original_matrix,original_meta,original_nodes,original_interactions,df,original_Omnipath_nodes,original_Omnipath_interactions))
    print("Day 0")
    rep_1_list.append(get_cell_LR_embeddings(first_matrix,first_meta,first_nodes,first_interactions,df,first_Omnipath_nodes,first_Omnipath_interactions))
    print("Done Rep 1")
    rep_2_list.append(get_cell_LR_embeddings(second_matrix,second_meta,second_nodes,second_interactions,df,second_Omnipath_nodes,second_Omnipath_interactions))
    print("Done Rep 2")

Omnipath_database = pd.read_csv("/data/LR_database/intercell_Omnipath.csv",index_col=0)
Omnipath_database.columns = ["from","to","references"]
os.system("mkdir -p /results/Pre_Post")
new_day_0_list = []
for i in day_0_list:
    new_day_0_list.append(pd.merge(i,Omnipath_database,left_on=["Src","Dst"],right_on=["from","to"])[["Src","Dst"]])
new_rep_1_list = []
for i in rep_1_list:
    new_rep_1_list.append(pd.merge(i,Omnipath_database,left_on=["Src","Dst"],right_on=["from","to"])[["Src","Dst"]])
new_rep_2_list = []
for i in rep_2_list:
    new_rep_2_list.append(pd.merge(i,Omnipath_database,left_on=["Src","Dst"],right_on=["from","to"])[["Src","Dst"]])
for i in range(5):
    if i % 10 == 0:
        print(i)
    new_day_0_list[i].drop_duplicates().head(1500).to_csv(f"/results/Pre_Post/Day0_{i+96}_random.csv") 
    new_rep_1_list[i].drop_duplicates().head(1500).to_csv(f"/results/Pre_Post/Rep1_{i+96}_random.csv")
    new_rep_2_list[i].drop_duplicates().head(1500).to_csv(f"/results/Pre_Post/Rep2_{i+96}_random.csv")
                                                    



