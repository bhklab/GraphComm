#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import liana as li
import anndata
import scanpy as sc
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


# # Pre-processing from original matrix
# 

# In[104]:


adata = sc.read_h5ad("/data/raw_data/Cardiac_cells/Visium-FZ_GT_P19.h5ad")
sc.pp.log1p(adata)


# In[105]:


matrix = pd.DataFrame.sparse.from_spmatrix(adata.X,index=adata.obs.index.tolist(),columns=adata.var["feature_name"].tolist())


# In[106]:


meta = pd.read_csv("/data/GraphComm_Input/Cardiac_cells/meta.csv",index_col=0)
nodes = pd.read_csv("/data/GraphComm_Input/Cardiac_cells/nodes.csv",index_col=0)
interactions = pd.read_csv("/data/GraphComm_Input/Cardiac_cells/interactions.csv",index_col=0)


# In[107]:


matrix = matrix.transpose()


# In[108]:


matrix = matrix[meta.index.tolist()]


# In[109]:


from scipy.stats import wilcoxon
import anndata
import scanpy as sc


# In[110]:


index = matrix.index.tolist()


# # Get Omnipath embedding

# In[111]:


LR_nodes = pd.read_csv("/data/LR_database/new_OmniPath_nodes.csv",index_col=0)
Omnipath_network = pd.read_csv("/data/LR_database/new_OmniPath_interactions.csv",index_col=0)
new_identifier = [row["identifier"] + "_" + row["category"] for index,row in LR_nodes.iterrows()]
LR_nodes["identifier"] = new_identifier
LR_nodes.index = LR_nodes["Id"].tolist()


# In[112]:


Omnipath_network["Src"] = LR_nodes.loc[Omnipath_network["Src"].tolist()]["identifier"].tolist()
Omnipath_network["Dst"] = LR_nodes.loc[Omnipath_network["Dst"].tolist()]["identifier"].tolist()


# In[113]:


ligand_list = nodes[nodes["category"]=="Ligand"]['identifier'].tolist()
receptor_list = nodes[nodes["category"]=="Receptor"]['identifier'].tolist()


# In[114]:


LR_nodes = LR_nodes[(LR_nodes["identifier"].isin(ligand_list)) | (LR_nodes["identifier"].isin(receptor_list))]


# In[115]:


Omnipath_network = Omnipath_network[(Omnipath_network["Src"].isin(LR_nodes["identifier"].tolist())) & (Omnipath_network["Dst"].isin(LR_nodes["identifier"].tolist()))]


# In[116]:


LR_nodes["Id"] = range(0,LR_nodes.shape[0])


# In[117]:


LR_nodes.index = LR_nodes["identifier"].tolist()
Omnipath_network["Src"] = LR_nodes.loc[Omnipath_network["Src"].tolist()]["Id"].tolist()
Omnipath_network["Dst"] = LR_nodes.loc[Omnipath_network["Dst"].tolist()]["Id"].tolist()


# In[118]:


LR_nodes = LR_nodes[(LR_nodes["Id"].isin(Omnipath_network["Src"].tolist())) | (LR_nodes["Id"].isin(Omnipath_network["Dst"].tolist()))]


# In[119]:


LR_nodes.index = LR_nodes["Id"].tolist()
Omnipath_network["Src"] = LR_nodes.loc[Omnipath_network["Src"].tolist()]["identifier"].tolist()
Omnipath_network["Dst"] = LR_nodes.loc[Omnipath_network["Dst"].tolist()]["identifier"].tolist()


# In[120]:


LR_nodes.index = LR_nodes["identifier"].tolist()
LR_nodes["Id"] = range(0,LR_nodes.shape[0])
Omnipath_network["Src"] = LR_nodes.loc[Omnipath_network["Src"].tolist()]["Id"].tolist()
Omnipath_network["Dst"] = LR_nodes.loc[Omnipath_network["Dst"].tolist()]["Id"].tolist()


# In[ ]:


Omnipath_data,Omnipath_nodes,Omnipath_interactions = make_dataset(LR_nodes,Omnipath_network,first=False,pathway_encode=False)


# In[ ]:


node_info = pd.DataFrame(np.zeros((Omnipath_nodes.shape[0],Omnipath_nodes.shape[0])),index=Omnipath_nodes["identifier"].tolist(),columns=Omnipath_nodes["identifier"].tolist())


# In[ ]:


temp_identifiers = [i.split("_")[0] for i in Omnipath_nodes["identifier"].tolist()]


# In[ ]:


complexes = pd.read_csv("/data/LR_database/complexes.csv")
complexes = complexes[complexes["member"].isin(temp_identifiers)]


# In[40]:


temp_nodes = Omnipath_nodes.copy()
temp_nodes.index = temp_identifiers
temp_nodes = temp_nodes[~temp_nodes.index.duplicated(keep='first')]


# In[41]:


complexes["member"] = temp_nodes.loc[complexes["member"].tolist()]["identifier"].tolist()


# In[42]:


group_complex = complexes.groupby("complex").agg(list)


# In[43]:


group_complex.index=range(0,group_complex.shape[0])


# In[44]:


for index,row in group_complex.iterrows():
    node_info.loc[list(set(row["member"])),list(set(row["member"]))] = index


# In[45]:


# for i in group_complex["member"].tolist():
#     node_info.loc[list(set(i)),list(set(i))] = 1


# In[47]:


pathways = pd.read_csv("/data/LR_database/kegg_pathways.csv",index_col=0)
pathways = pathways[pathways["genesymbol"].isin(temp_identifiers)]
pathways["genesymbol"] = temp_nodes.loc[pathways["genesymbol"].tolist()]["identifier"].tolist()
group_pathway = pathways.groupby("pathway").agg(list)


# In[48]:


group_pathway.index=range(0,group_pathway.shape[0])


# In[49]:


for index,row in group_pathway.iterrows():
    node_info.loc[list(set(row["genesymbol"])),list(set(row["genesymbol"]))] += index


# In[50]:


# for i in group_pathway["genesymbol"].tolist():
#     node_info.loc[list(set(i)),list(set(i))] += 1


# In[51]:


truth_info = pd.DataFrame(np.zeros((Omnipath_nodes.shape[0],Omnipath_nodes.shape[0])),index=Omnipath_nodes["identifier"].tolist(),columns=Omnipath_nodes["identifier"].tolist())


# In[52]:


Omnipath_nodes.index = Omnipath_nodes["Id"].tolist()


# In[53]:


ident_interactions = Omnipath_interactions.copy()
ident_interactions["Src"] = Omnipath_nodes.loc[ident_interactions["Src"].tolist()]["identifier"].tolist()
ident_interactions["Dst"] = Omnipath_nodes.loc[ident_interactions["Dst"].tolist()]["identifier"].tolist()


# In[54]:


for index,row in ident_interactions.iterrows():
    truth_info.loc[row["Src"],row["Dst"]] = 1


# In[55]:


ligands = Omnipath_nodes[Omnipath_nodes["category"]=="Ligand"]["identifier"].tolist()
receptors = Omnipath_nodes[Omnipath_nodes["category"]=="Receptor"]["identifier"].tolist()
#truth_info = truth_info.loc[ligands,receptors]
truth_info = torch.Tensor(truth_info.values).to(device)


# In[56]:


ligands = Omnipath_nodes[Omnipath_nodes["category"]=="Ligand"]["Id"].tolist()
receptors = Omnipath_nodes[Omnipath_nodes["category"]=="Receptor"]["Id"].tolist()


# In[57]:


ident_interactions = ident_interactions.drop_duplicates("Src")
ident_interactions = ident_interactions.drop_duplicates("Dst")


# In[58]:


ident_interactions.index = range(0,ident_interactions.shape[0])


# In[59]:


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


# In[60]:


node_info.values[np.where(np.isnan(node_info.values))] = 0
node_info.values[np.where(np.isinf(node_info.values))] = 0


# In[61]:


Omnipath_data.x = torch.Tensor(node_info.values)


# In[62]:


#Omnipath_data.y = torch.Tensor(truth_list).type(torch.LongTensor)
Omnipath_data.y = truth_info


# In[63]:


Omnipath_nodes.index = Omnipath_nodes["Id"].tolist()


# In[64]:


Omnipath_interactions["Src"] = [Omnipath_nodes.loc[i]["identifier"] for i in Omnipath_interactions["Src"].tolist()]
Omnipath_interactions["Dst"] = [Omnipath_nodes.loc[i]["identifier"] for i in Omnipath_interactions["Dst"].tolist()]


# # Random Walks

# In[66]:


from torch_geometric.nn import Node2Vec
import os.path as osp
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.datasets import Planetoid
from tqdm.notebook import tqdm


# In[67]:


data = Omnipath_data


# ## data loader

# In[ ]:


model = Omnipath_Node2Vec(data.edge_index, embedding_dim=2, walk_length=40,
                 context_size=40, walks_per_node=10,
                 num_negative_samples=1, p=1, q=1, sparse=True).to(device)

loader = model.loader(batch_size=2, shuffle=True, num_workers=4)
optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)






print("Getting embeddings from ground truth")

for epoch in range(1, 100):
    loss = Omnipath_train(model,loader,optimizer)
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')




model.eval()
z = model(torch.arange(data.num_nodes)).detach()


# In[74]:


ligand_ids = Omnipath_nodes[Omnipath_nodes["category"].str.contains("Ligand")]["Id"].tolist()
receptor_ids = Omnipath_nodes[Omnipath_nodes["category"].str.contains("Receptor")]["Id"].tolist()


# In[75]:


ligand_embeddings = z[ligand_ids,:]
receptor_embeddings = z[receptor_ids,:]


# In[76]:


total_embeddings = torch.inner(ligand_embeddings,receptor_embeddings)


# In[77]:


total_embeddings_df = pd.DataFrame(total_embeddings.numpy(),index=Omnipath_nodes[Omnipath_nodes["category"].str.contains("Ligand")]["identifier"].tolist(),columns=Omnipath_nodes[Omnipath_nodes["category"].str.contains("Receptor")]["identifier"].tolist())


# # make cell -> ligand and receptor graphs 

# In[121]:


cell_LR_data,cell_LR_nodes,cell_LR_ints = make_dataset(nodes,interactions,first=False,pathway_encode=False)


# In[218]:


full_matrix = pd.DataFrame(np.zeros((cell_LR_nodes.shape[0],cell_LR_nodes.shape[0])),index=cell_LR_nodes["identifier"].tolist(),columns=cell_LR_nodes["identifier"].tolist())


# In[219]:


ligands = Omnipath_nodes[Omnipath_nodes["category"]=="Ligand"]["Id"].tolist()
receptors = Omnipath_nodes[Omnipath_nodes["category"]=="Receptor"]["Id"].tolist()


# In[220]:


total_out_df = total_embeddings_df


# In[221]:


#matrix = matrix.transpose()


# In[222]:


gene_mean = matrix.mean(axis=1)


# In[223]:


Omnipath_nodes.index = [i.split("_")[0] for i in Omnipath_nodes["identifier"].tolist()]


# In[224]:


Omnipath_nodes = Omnipath_nodes.loc[~Omnipath_nodes.index.duplicated(),:].copy()


# In[225]:


gene_mean = gene_mean.loc[Omnipath_nodes.index.tolist()]


# In[226]:


gene_mean.index = Omnipath_nodes["identifier"].tolist()


# In[227]:


ligands = [i for i in gene_mean.index.tolist() if "Ligand" in i]
receptors = [i for i in gene_mean.index.tolist() if "Receptor" in i]
ligands = list(set(full_matrix.index.tolist()) & set(ligands))
receptors = list(set(full_matrix.index.tolist()) & set(receptors))


# In[228]:


for i,j in zip(ligands,receptors):
    if (i in gene_mean.index.tolist()) and (j in gene_mean.index.tolist()):
        full_matrix.loc[i,j] = gene_mean.loc[i]*gene_mean.loc[j]


# In[229]:


ligands = list(set(ligands) & set(total_out_df.index.tolist()))
receptors = list(set(receptors) & set(total_out_df.columns.tolist()))


# In[230]:


for i,j in zip(ligands,receptors):
    full_matrix.loc[i,j] += total_out_df.loc[i,j]
    


# In[231]:


#full_matrix.loc[true_df.index.tolist(),true_df.columns.tolist()] = true_df.values


# In[250]:


cell_groups = meta["labels"].unique().tolist()
cell_groups = [i for i in cell_groups if i != "Lymphoid"]


# In[234]:


ligands = cell_LR_nodes[cell_LR_nodes["category"]=="Ligand"]["identifier"].tolist()
ligands = [i.split("_")[0] for i in ligands]
ligand_matrix = matrix.loc[ligands]


# In[235]:


mean_dict = {}
for i in cell_groups:
    cells = meta[meta["labels"]==i]["cell"].tolist()
    mean_dict[i] = ligand_matrix[cells].mean(axis=1)


# In[236]:


for i in mean_dict.keys():
    temp_index = [i+"_Ligand" for i in mean_dict[i].index.tolist()]
    full_matrix.loc[i,temp_index] = mean_dict[i].values
    full_matrix.loc[temp_index,i] = mean_dict[i].values


# In[238]:


receptors = cell_LR_nodes[cell_LR_nodes["category"]=="Receptor"]["identifier"].tolist()
receptors = [i.split("_")[0] for i in receptors]
receptors_matrix = matrix.loc[receptors]


# In[239]:


mean_dict = {}
for i in cell_groups:
    cells = meta[meta["labels"]==i]["cell"].tolist()
    mean_dict[i] = receptors_matrix[cells].mean(axis=1)


# In[240]:


for i in mean_dict.keys():
    temp_index = [i+"_Receptor" for i in mean_dict[i].index.tolist()]
    full_matrix.loc[i,temp_index] = mean_dict[i].values
    full_matrix.loc[temp_index,i] = mean_dict[i].values


# In[242]:


full_matrix.values[np.where(np.isnan(full_matrix.values))] = 0
full_matrix.values[np.where(np.isinf(full_matrix.values))] = 0


# In[244]:


adata = sc.read_h5ad("/data/raw_data/Cardiac_cells/Visium-FZ_GT_P19.h5ad")
adata = adata[meta.index.tolist()]


# In[245]:


spatial_coordinates = adata.obsm["X_spatial"]


# In[246]:


spatial_df = pd.DataFrame({"x":spatial_coordinates[:,0],"y":spatial_coordinates[:,1]},index=meta.index.tolist())


# In[251]:


import math


# In[308]:


spatial_dict = {}
for i in cell_groups:
    cells = meta[meta["labels"]==i].index.tolist()
    spatial_coords = [(i,j) for i,j in zip(spatial_df.loc[cells]["x"].tolist(),spatial_df.loc[cells]["y"].tolist())]
    for j in cell_groups:
        if j != i:
            second_cells = meta[meta["labels"]==j].index.tolist()
            second_spatial_coords = [(i,j) for i,j in zip(spatial_df.loc[second_cells]["x"].tolist(),spatial_df.loc[second_cells]["y"].tolist())]
            min_list = []
            for k in spatial_coords:
                min_list.append(min([math.dist(k,l) for l in second_spatial_coords]))
            spatial_dict[(i,j)] = min(min_list)


# In[309]:


for k in spatial_dict.keys():
    full_matrix.loc[k[0],k[1]] = spatial_dict[k]
    full_matrix.loc[k[1],k[0]] = spatial_dict[k]


# In[326]:


full_matrix = full_matrix.drop("Lymphoid")
full_matrix = full_matrix.drop("Lymphoid",axis=1)


# In[327]:


cell_LR_data.x = torch.Tensor(full_matrix.values)


# In[328]:


LR_ids = cell_LR_nodes[(cell_LR_nodes["category"]=="Ligand") | (cell_LR_nodes["category"]=="Receptor")]["Id"].tolist()


# In[329]:


# true_values["Src"] = [i + "_Ligand" for i in true_values["Src"].tolist()]
# true_values["Dst"] = [i + "_Receptor" for i in true_values["Dst"].tolist()]


# In[330]:


truth_list = []
for i in cell_LR_nodes["identifier"].tolist():
    if "Ligand" in i:
        if i in ident_interactions["Src"].tolist():
            #truth_list.append(ident_interactions[ident_interactions["Src"]==i].index.tolist()[0] + 1)
            truth_list.append(1)
        else:
            truth_list.append(0)
    elif "Receptor" in i:
        if i in ident_interactions["Dst"].tolist():
            #truth_list.append(ident_interactions[ident_interactions["Dst"]==i].index.tolist()[0] + 1)
            truth_list.append(1)
        else:
            truth_list.append(0)
    else:
        truth_list.append(2)


# In[331]:


cell_LR_data.y = torch.Tensor(truth_list).type(torch.LongTensor)


# In[332]:


truth_array = np.array(truth_list)
positive_classes = np.where(truth_array==1)[0].tolist()
negative_classes = np.where(truth_array==0)[0].tolist()[:len(positive_classes)]


# In[333]:


new_train_mask = np.array([False]*truth_array.shape[0])
new_train_mask[positive_classes + negative_classes] = True


# In[334]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"

model = GAT(cell_LR_data,num_classes=2).to(device)
data = cell_LR_data.to(device)


# In[335]:


data.train_mask = torch.Tensor(new_train_mask).type(torch.LongTensor)


# In[336]:


optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()


# In[337]:


ligands = cell_LR_nodes[cell_LR_nodes["category"]=="Ligand"]["Id"].tolist()
receptors = cell_LR_nodes[cell_LR_nodes["category"]=="Receptor"]["Id"].tolist()


# In[338]:


truth_df = full_matrix.loc[cell_LR_nodes[cell_LR_nodes["category"]=="Ligand"]["identifier"].tolist(),cell_LR_nodes[cell_LR_nodes["category"]=="Receptor"]["identifier"].tolist()]


# In[339]:


truth_Tensor = torch.Tensor(truth_df.values).to(device)


# In[340]:


for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[new_train_mask],data.y[new_train_mask])
    ligand_out = out[ligands,:]
    receptor_out = out[receptors,:]
    total_out = torch.inner(ligand_out,receptor_out)
    #loss = criterion(out[LR_ids],data.y)
    #loss = criterion(total_out,truth_Tensor)
    #loss = criterion(out[new_train_mask],data.y[new_train_mask])
    if epoch%20 == 0:
        print(loss)
    
    loss.backward()
    optimizer.step()
    


# In[372]:


model.eval()
cell_LR_out = model(data)
out = cell_LR_out


# In[373]:


ligands = cell_LR_nodes[cell_LR_nodes["category"]=="Ligand"]["Id"].tolist()
receptors = cell_LR_nodes[cell_LR_nodes["category"]=="Receptor"]["Id"].tolist()


# In[374]:


ligand_out = cell_LR_out[ligands,:]
receptor_out = cell_LR_out[receptors,:]
_,ligand_pred = ligand_out.max(dim=1)
_,receptor_pred = receptor_out.max(dim=1)

total_out = torch.inner(ligand_out,receptor_out).cpu().detach().numpy()


# In[375]:


cell_LR_nodes.index = cell_LR_nodes["Id"].tolist()


# In[376]:


ligand_df = cell_LR_nodes[cell_LR_nodes["category"]=="Ligand"]
receptor_df = cell_LR_nodes[cell_LR_nodes["category"]=="Receptor"]
ligand_df.index = range(ligand_df.shape[0])
receptor_df.index = range(receptor_df.shape[0])


# In[377]:


valid_ligands = ligand_df
valid_receptors = receptor_df


# In[378]:


ligand_pred = ligand_pred.cpu().detach().numpy()
receptor_pred = receptor_pred.cpu().detach().numpy()


# In[379]:


ligand_out = ligand_out[valid_ligands.index.tolist()]
receptor_out = receptor_out[valid_receptors.index.tolist()]


# In[380]:


ligand_nodes = cell_LR_nodes[cell_LR_nodes["category"] == "Ligand"]
ligand_nodes.index = range(0,ligand_nodes.shape[0])
ligand_idents = ligand_nodes.iloc[np.where(ligand_pred==1)]['identifier'].tolist()


# In[381]:


total_out_df = pd.DataFrame(total_out,index=valid_ligands["identifier"].tolist(),columns=valid_receptors["identifier"].tolist())


# In[382]:


indicies = np.where(total_out_df.values > 0)
source = list(indicies[0])
dest = list(indicies[1])


# In[383]:


index_df = pd.DataFrame({"Id":range(0,total_out_df.shape[0]),"identifier":total_out_df.index.tolist()})
column_df = pd.DataFrame({"Id":range(0,total_out_df.shape[1]),"identifier":total_out_df.columns.tolist()})


# In[384]:


source_list = index_df.loc[source]["identifier"].tolist()
dest_list = column_df.loc[dest]["identifier"].tolist()


# In[385]:


total_link_df = pd.DataFrame({"Src":source_list,"Dst":dest_list,"Prob":total_out_df.values[indicies]})


# In[386]:


total_link_df = total_link_df.sort_values("Prob",ascending=False)


# In[387]:


Omnipath_db = pd.read_csv("/data/LR_database/new_Omnipath_database.csv")


# In[388]:


total_link_df["Src"] = [i.split("_")[0] for i in total_link_df["Src"].tolist()]
total_link_df["Dst"] = [i.split("_")[0] for i in total_link_df["Dst"].tolist()]


# In[389]:


total_link_df = total_link_df.drop_duplicates()


# In[390]:


Omnipath_db = Omnipath_db.drop_duplicates(["from","to"])


# In[391]:


LR_out = cell_LR_out[valid_ligands["Id"].tolist() + valid_receptors["Id"].tolist(),:]
cell_group_out = cell_LR_out[cell_LR_nodes[cell_LR_nodes["category"]=="Cell Group"]["Id"].tolist(),:]
cell_LR_out = torch.inner(LR_out,cell_group_out).cpu().detach().numpy()


# In[392]:


cell_LR_df = pd.DataFrame(cell_LR_out,index=valid_ligands["identifier"].tolist() + valid_receptors["identifier"].tolist(),columns=cell_LR_nodes[cell_LR_nodes["category"]=="Cell Group"]["identifier"].tolist())


# In[393]:


ligands = nodes[nodes["category"]=="Ligand"]["identifier"].tolist()
receptors = nodes[nodes["category"]=="Receptor"]["identifier"].tolist()
expression_df = matrix


# In[394]:


cell_groups = meta["labels"].unique().tolist()
mean_matrix = pd.DataFrame(columns=cell_groups,index=ligands+receptors)
for i in cell_groups:
    cells = meta[meta["labels"]==i]["cell"].tolist()
    temp_ligands = [i.split("_")[0] for i in ligands]
    ligand_df = expression_df[cells].mean(axis=1).loc[temp_ligands]
    ligand_df.index = [i+"_Ligand" for i in ligand_df.index.tolist()]
    temp_receptors = receptors = [i.split("_")[0] for i in receptors]
    receptor_df = expression_df[cells].mean(axis=1).loc[temp_receptors]
    receptor_df.index = [i+"_Receptor" for i in receptor_df.index.tolist()]
    total_df = pd.concat([ligand_df,receptor_df])
    mean_matrix[i] = total_df.tolist()


# In[395]:


ligands = cell_LR_nodes[cell_LR_nodes["category"]=="Ligand"]["Id"].tolist()
receptors = cell_LR_nodes[cell_LR_nodes["category"]=="Receptor"]["Id"].tolist()


# In[396]:


interacting_ligands = list(set(ligands) & set(cell_LR_ints["Dst"].tolist()))
interacting_receptors = list(set(receptors) & set(cell_LR_ints["Dst"].tolist()))


# In[99]:


cell_LR_ints.index = cell_LR_ints["Dst"].tolist()
ligand_cells = cell_LR_ints.loc[interacting_ligands]["Src"].unique().tolist()
receptor_cells = cell_LR_ints.loc[interacting_receptors]["Src"].unique().tolist()


# In[397]:


cell_LR_out = torch.Tensor(cell_LR_out)


# In[398]:


ligand_cell_out = out[ligand_cells,:]
ligand_out = out[ligands,:]
total_ligand_out = torch.inner(ligand_out,ligand_cell_out).cpu().detach().numpy()
receptor_cell_out = out[receptor_cells,:]
receptor_out = out[receptors,:]
total_receptor_out = torch.inner(receptor_out,receptor_cell_out).cpu().detach().numpy()


# In[400]:


mean_matrix = mean_matrix.drop("Lymphoid",axis=1)


# In[409]:


np.max(matrix.values)


# In[402]:


ligand_matrix = mean_matrix[mean_matrix.index.str.contains("Ligand")]
receptor_matrix = mean_matrix[mean_matrix.index.str.contains("Receptor")]


# In[112]:


ligand_cell_out = np.multiply(ligand_matrix,total_ligand_out)
receptor_cell_out = np.multiply(receptor_matrix,total_receptor_out)


# In[113]:


ligand_maxes = (ligand_cell_out.idxmax(axis=1))
receptor_maxes = (receptor_cell_out.idxmax(axis=1))


# In[403]:


ligand_maxes.index = [i.split("_")[0] for i in ligand_maxes.index.tolist()]
receptor_maxes.index = [i.split("_")[0] for i in receptor_maxes.index.tolist()]


# In[404]:


total_link_df["Src Cell"] = ligand_maxes.loc[total_link_df['Src']].tolist()
total_link_df["Dst Cell"] = receptor_maxes.loc[total_link_df['Dst']].tolist()


# In[406]:


total_link_df.to_csv("/data/GraphComm_Output/Cardiac_cells/CCI.csv")

