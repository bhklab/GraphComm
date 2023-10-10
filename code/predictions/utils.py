




import os
import torch
# os.environ['TORCH'] = torch.__version__
# os.system('pip install -q torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html')
# os.system('pip install -q torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html')
# os.system('pip install -q git+https://github.com/pyg-team/pytorch_geometric.git')


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from utils import *
import argparse
import os.path as osp
import umap 
import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import scanpy as sc
import torch
from torch import Tensor
from torch.nn import Embedding
from torch.utils.data import DataLoader

from torch_geometric.utils import sort_edge_index
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.sparse import index2ptr

try:
    import torch_cluster  # noqa
    random_walk = torch.ops.torch_cluster.random_walk
except ImportError:
    random_walk = None
from torch_geometric.datasets import Entities
from torch_geometric.nn import FastRGCNConv, RGCNConv, GCNConv, InnerProductDecoder, GAE, VGAE
from torch_geometric.utils import k_hop_subgraph

from torch_geometric.datasets import Planetoid
from torch_geometric.data.data import Data
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GATConv
import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Parameter
from tqdm import tqdm
from torch_geometric.nn import GAE, RGCNConv

from sklearn.model_selection import train_test_split
from torch_geometric.data import Dataset

from torch_geometric.nn import Node2Vec
import os.path as osp
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.datasets import Planetoid
from tqdm.notebook import tqdm

import torch_cluster

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
from torch_geometric.nn import GATConv

from torch_geometric.data import Data 

from scipy.spatial.distance import squareform, pdist
import anndata
from model import *
def make_dataset(nodes,interactions,first=True,pathway_encode=False):
    data = Data()
    if first:
        interactions["Src"] = interactions["Src"] -1 
        interactions["Dst"] = interactions["Dst"] -1 
        nodes["Id"] = nodes["Id"] - 1

    edge_index = np.transpose(interactions.iloc[:,:2].to_numpy())
    x = nodes["Id"].to_numpy()
    x = np.array(np.transpose(np.matrix(x)))
    data = Data(x=(torch.tensor(x)).float(),edge_index=torch.tensor(edge_index))
    train_mask = [True]*int(np.floor(data.num_nodes/2)) + [False]* int(np.ceil(data.num_nodes/2))
    test_mask = [False]*int(np.floor(data.num_nodes/2)) + [True]* int(np.ceil(data.num_nodes/2))
    data.train_mask = torch.tensor(train_mask)
    data.test_mask = torch.tensor(test_mask)
    data.edge_type = torch.tensor(interactions["edge_type"].tolist())
    data.num_nodes = nodes.shape[0]

    y_dict = {"Cell Group": 0,"Cell":1,"Ligand":2,"Receptor":3}
    categories = nodes["category"].tolist()
    new_y = []
    for i in categories:
        new_y.append(y_dict[i])
    data.y = torch.tensor(new_y)

    y_dataframe = pd.DataFrame({"Id":nodes["Id"].tolist(),"y":new_y})
    cell_group = y_dataframe[y_dataframe["y"] == 0]
    cells_df = y_dataframe[y_dataframe["y"] == 1]
    ligand_df = y_dataframe[y_dataframe["y"] == 2]
    receptor_df = y_dataframe[y_dataframe["y"] == 3]
    train_df = pd.concat([cell_group,cells_df,ligand_df,receptor_df])
    data.train_idx = torch.tensor(train_df["Id"].tolist())
    data.train_y = torch.tensor(train_df["y"].tolist())
    test_df = pd.concat([cell_group,cells_df,ligand_df,receptor_df])
    data.test_idx = torch.tensor(test_df["Id"].tolist())
    data.test_y = torch.tensor(test_df["y"].tolist())

    edge_type_df = interactions["edge_type"].tolist()
    edge_df = interactions.iloc[:,:2]
    train_edge_index = edge_df
    test_edge_index = edge_df
    val_edge_index = edge_df
    train_edge_type = edge_type_df
    test_edge_type = edge_type_df
    val_edge_type = edge_type_df
    #train_edge_index,test_edge_index,train_edge_type,test_edge_type = train_test_split(edge_df,edge_type_df,test_size=0.0)
    #train_edge_index,val_edge_index,train_edge_type,val_edge_type = train_test_split(train_edge_index,train_edge_type,test_size=0.0)
    data.train_edge_index = torch.tensor(np.transpose(train_edge_index.to_numpy()))
    data.valid_edge_index = torch.tensor(np.transpose(val_edge_index.to_numpy()))
    data.test_edge_index = torch.tensor(np.transpose(test_edge_index.to_numpy()))
    data.train_edge_type = torch.tensor(train_edge_type)
    data.valid_edge_type = torch.tensor(val_edge_type)
    data.test_edge_type = torch.tensor(test_edge_type)
    if pathway_encode:
        pathways = pd.read_csv("/data/kegg_pathways.csv",index_col=0)
        available_ligands = nodes[nodes['category']=="Ligand"]["identifier"].tolist()
        pathway_ligands = pathways[pathways["genesymbol"].isin(available_ligands)]["genesymbol"].tolist()
        pathway_names = pathways[pathways["genesymbol"].isin(available_ligands)].groupby(["pathway"]).count()["genesymbol"]
        top_pathways = pathway_names[pathway_names > 0].sort_values(ascending=False).index.tolist()
        pathway_df = pd.DataFrame()
        for i in top_pathways:
            pathway_df[i] = get_pathway_encodings(pathways,i,nodes,interactions)
        data.x = torch.tensor(pathway_df.values)
    return data,nodes,interactions
    
def get_Omnipath_embeddings(nodes,interactions,reproduce=None,save=None,lr=0.01):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = "cpu"
    torch.manual_seed(0)
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
        node_info.loc[list(set(row["member"])),list(set(row["member"]))] += 1
        
    pathways = pd.read_csv("/data/LR_database/kegg_pathways.csv",index_col=0)
    pathways = pathways[pathways["genesymbol"].isin(temp_identifiers)]
    pathways["genesymbol"] = temp_nodes.loc[pathways["genesymbol"].tolist()]["identifier"].tolist()
    group_pathway = pathways.groupby("pathway").agg(list)

    group_pathway.index=range(0,group_pathway.shape[0])

    for index,row in group_pathway.iterrows():
        node_info.loc[list(set(row["genesymbol"])),list(set(row["genesymbol"]))] += 1
    truth_info = pd.DataFrame(np.zeros((Omnipath_nodes.shape[0],Omnipath_nodes.shape[0])),index=Omnipath_nodes["identifier"].tolist(),columns=Omnipath_nodes["identifier"].tolist())

    Omnipath_nodes.index = Omnipath_nodes["Id"].tolist()

    ident_interactions = Omnipath_interactions.copy()
    ident_interactions["Src"] = Omnipath_nodes.loc[ident_interactions["Src"].tolist()]["identifier"].tolist()
    ident_interactions["Dst"] = Omnipath_nodes.loc[ident_interactions["Dst"].tolist()]["identifier"].tolist()

    # for index,row in ident_interactions.iterrows():
    #     truth_info.loc[row["Src"],row["Dst"]] = 1

    ligands = Omnipath_nodes[Omnipath_nodes["category"]=="Ligand"]["identifier"].tolist()
    receptors = Omnipath_nodes[Omnipath_nodes["category"]=="Receptor"]["identifier"].tolist()
    truth_info = torch.Tensor(truth_info.values).to(device)


    ligands = Omnipath_nodes[Omnipath_nodes["category"]=="Ligand"]["Id"].tolist()
    receptors = Omnipath_nodes[Omnipath_nodes["category"]=="Receptor"]["Id"].tolist()


    ident_interactions = ident_interactions.drop_duplicates("Src")
    ident_interactions = ident_interactions.drop_duplicates("Dst")

    ident_interactions.index = range(0,ident_interactions.shape[0])
    
    node_info.values[np.where(np.isnan(node_info.values))] = 0
    node_info.values[np.where(np.isinf(node_info.values))] = 0

    Omnipath_data.x = torch.Tensor(node_info.values)

    Omnipath_nodes.index = Omnipath_nodes["Id"].tolist()

    Omnipath_interactions["Src"] = [Omnipath_nodes.loc[i]["identifier"] for i in Omnipath_interactions["Src"].tolist()]
    Omnipath_interactions["Dst"] = [Omnipath_nodes.loc[i]["identifier"] for i in Omnipath_interactions["Dst"].tolist()]
    
    edge_weights = [1 for i,j in zip(Omnipath_interactions["Src"].tolist(),Omnipath_interactions["Dst"].tolist())]
    data = Omnipath_data
    torch.manual_seed(0)
    if reproduce is not None:
        print("Reproducing results from original paper - no training process")
        model = torch.load(f"/data/models/{reproduce}/Omnipath.pt")
    else:
        model = Omnipath_Node2Vec(data.edge_index, embedding_dim=128, walk_length=10,
                     context_size=5, walks_per_node=10,
                     num_negative_samples=1, p=1, q=1, sparse=True).to(device)

        loader = model.loader(batch_size=2, shuffle=True, num_workers=4)
        optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=lr)
        for epoch in range(100):
            loss = Omnipath_train(model,loader,optimizer)
            if epoch % 10 == 0:
                print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
        os.system(f"mkdir -p //data/models/{save}")
        torch.save(model,f"//data/models/{save}/Omnipath.pt")
    model.eval()
    z = model(torch.arange(data.num_nodes)).detach()

    ligand_ids = Omnipath_nodes[Omnipath_nodes["category"] == "Ligand"]["Id"].tolist()
    receptor_ids = Omnipath_nodes[Omnipath_nodes["category"] == "Receptor"]["Id"].tolist()

    ligand_embeddings = z[ligand_ids,:]
    receptor_embeddings = z[receptor_ids,:]

    total_embeddings = torch.inner(ligand_embeddings,receptor_embeddings)
    ligand_data_x = data.x[ligand_ids]
    LR_data_x = ligand_data_x[:,receptor_ids]
    edited_values = torch.mul(total_embeddings[np.where(LR_data_x >= 0)],LR_data_x[np.where(LR_data_x >= 0)])
    total_embeddings[np.where(LR_data_x >= 0)] = edited_values
    total_embeddings_df = pd.DataFrame(total_embeddings.numpy(),index=Omnipath_nodes[Omnipath_nodes["category"].str.contains("Ligand")]["identifier"].tolist(),columns=Omnipath_nodes[Omnipath_nodes["category"].str.contains("Receptor")]["identifier"].tolist())
    return total_embeddings_df

def get_cell_LR_embeddings(matrix,meta,nodes,interactions,total_embeddings_df,Omnipath_nodes,Omnipath_interactions,spatial=None,reproduce=None,save=None):
    device = 'cpu'
    torch.manual_seed(0)
    truth_info = pd.DataFrame(np.zeros((Omnipath_nodes.shape[0],Omnipath_nodes.shape[0])),index=Omnipath_nodes["identifier"].tolist(),columns=Omnipath_nodes["identifier"].tolist())
    ident_interactions = Omnipath_interactions.copy()

    ligands = Omnipath_nodes[Omnipath_nodes["category"]=="Ligand"]["identifier"].tolist()
    receptors = Omnipath_nodes[Omnipath_nodes["category"]=="Receptor"]["identifier"].tolist()
    truth_info = torch.Tensor(truth_info.values).to(device)

    
    cell_LR_data,cell_LR_nodes,cell_LR_ints = make_dataset(nodes,interactions,first=False,pathway_encode=False)

    full_matrix = pd.DataFrame(np.zeros((cell_LR_nodes.shape[0],cell_LR_nodes.shape[0])),index=cell_LR_nodes["identifier"].tolist(),columns=cell_LR_nodes["identifier"].tolist())

    ligands = Omnipath_nodes[Omnipath_nodes["category"]=="Ligand"]["Id"].tolist()
    receptors = Omnipath_nodes[Omnipath_nodes["category"]=="Receptor"]["Id"].tolist()

    total_out_df = total_embeddings_df

    gene_mean = matrix.mean(axis=1)

    Omnipath_nodes.index = [i.split("_")[0] for i in Omnipath_nodes["identifier"].tolist()]

    Omnipath_nodes = Omnipath_nodes.loc[~Omnipath_nodes.index.duplicated(),:].copy()
    

    gene_mean = gene_mean.loc[Omnipath_nodes.index.tolist()]
    gene_mean = gene_mean.loc[~gene_mean.index.duplicated()].copy()
    gene_mean.index = Omnipath_nodes["identifier"].tolist()

    ligands = [i for i in gene_mean.index.tolist() if "Ligand" in i]
    receptors = [i for i in gene_mean.index.tolist() if "Receptor" in i]
    ligands = list(set(full_matrix.index.tolist()) & set(ligands))
    receptors = list(set(full_matrix.index.tolist()) & set(receptors))
    ligands = list(set(ligands)&set(gene_mean.index.tolist()))
    receptors = list(set(receptors)&set(gene_mean.index.tolist()))
    new_ligands = []
    new_receptors = []
    print("Starting...")
    # for i in ligands:
    #     for j in receptors:
    #         new_ligands.append(i)
    #         new_receptors.append(j)
    # for i,j in zip(new_ligands,new_receptors):
    #         if (i in gene_mean.index.tolist()) and (j in gene_mean.index.tolist()):
    #             full_matrix.loc[i,j] = gene_mean.loc[i]*gene_mean.loc[j]
    for i in ligands:
        #print(type(gene_mean.loc[receptors].tolist()[0]))
       # print(type(gene_mean.loc[i]))
        full_matrix.loc[i][receptors] = (gene_mean.loc[receptors]*gene_mean.loc[i]).tolist()
    print("continuing...")
    ligands = list(set(ligands) & set(total_out_df.index.tolist()))
    receptors = list(set(receptors) & set(total_out_df.columns.tolist()))
    new_ligands = []
    new_receptors = []
    # for i in ligands:
    #     for j in receptors:
    #         new_ligands.append(i)
    #         new_receptors.append(j)
    # for i,j in zip(new_ligands,new_receptors):
    #         full_matrix.loc[i,j] *= total_out_df.loc[i,j]  
    for i in ligands:
        full_matrix.loc[i][receptors] *= total_out_df.loc[i][receptors]
    cell_groups = meta["labels"].unique().tolist()

    ligands = cell_LR_nodes[cell_LR_nodes["category"]=="Ligand"]["identifier"].tolist()
    ligands = [i.split("_")[0] for i in ligands]
    ligand_matrix = matrix.loc[ligands]

    ligand_matrix = ligand_matrix[~ligand_matrix.index.duplicated(keep='first')]


    mean_dict = {}
    for i in cell_groups:
        cells = meta[meta["labels"]==i]["cell"].tolist()
        mean_dict[str(i)] = ligand_matrix[cells].mean(axis=1)

    full_matrix = full_matrix[~full_matrix.index.duplicated(keep='first')]
    full_matrix = full_matrix.loc[:,~full_matrix.columns.duplicated()].copy()


    for i in mean_dict.keys():
        temp_index = [i+"_Ligand" for i in mean_dict[i].index.tolist()]
        full_matrix.loc[i,temp_index] = mean_dict[i].values
        full_matrix.loc[temp_index,i] = mean_dict[i].values

    receptors = cell_LR_nodes[cell_LR_nodes["category"]=="Receptor"]["identifier"].tolist()
    receptors = [i.split("_")[0] for i in receptors]
    receptors_matrix = matrix.loc[receptors]

    receptors_matrix = receptors_matrix[~receptors_matrix.index.duplicated(keep='first')]


    mean_dict = {}
    for i in cell_groups:
        cells = meta[meta["labels"]==i]["cell"].tolist()
        mean_dict[str(i)] = receptors_matrix[cells].mean(axis=1)

    full_matrix = full_matrix[~full_matrix.index.duplicated(keep='first')]
    full_matrix = full_matrix.loc[:,~full_matrix.columns.duplicated()].copy()


    for i in mean_dict.keys():
        temp_index = [i+"_Receptor" for i in mean_dict[i].index.tolist()]
        full_matrix.loc[i,temp_index] = mean_dict[i].values
        full_matrix.loc[temp_index,i] = mean_dict[i].values

    full_matrix.values[np.where(np.isnan(full_matrix.values))] = 0
    full_matrix.values[np.where(np.isinf(full_matrix.values))] = 0
    
    if spatial is not None:
        adata = sc.read_h5ad(spatial)
        adata = adata[meta.index.tolist()]

        spatial_coordinates = adata.obsm["X_spatial"]

        spatial_df = pd.DataFrame({"x":spatial_coordinates[:,0],"y":spatial_coordinates[:,1]},index=meta.index.tolist())

        cell_groups = meta["labels"].unique().tolist()

        import math

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


        for k in spatial_dict.keys():
            full_matrix.loc[k[0],k[1]] = spatial_dict[k]
            full_matrix.loc[k[1],k[0]] = spatial_dict[k]

    cell_LR_data.x = torch.Tensor(full_matrix.values)

    LR_ids = cell_LR_nodes[(cell_LR_nodes["category"]=="Ligand") | (cell_LR_nodes["category"]=="Receptor")]["Id"].tolist()

    cell_groups = cell_LR_nodes[cell_LR_nodes["category"]=="Cell Group"]['identifier'].tolist()
    #ident_interactions = pd.read_csv("/data/LR_database/intercell_Omnipath.csv",index_col=0)
    ident_interactions = pd.read_csv("//data/LR_database/consensus_Omnipath.csv",index_col=0)
    #ident_interactions.columns = ["Src","Dst","references"]
    ident_interactions.columns = ["Src","Dst"]
    ident_interactions["Src"] = [i+"_Ligand" for i in ident_interactions["Src"].tolist()]
    ident_interactions["Dst"] = [i+"_Receptor" for i in ident_interactions["Dst"].tolist()]
    
    truth_list = []
    for i in cell_LR_nodes["identifier"].tolist():
        if "Ligand" in i:
            if i in ident_interactions["Src"].tolist():
                truth_list.append(1)
            else:
                truth_list.append(0)
        elif "Receptor" in i:
            if i in ident_interactions["Dst"].tolist():
                truth_list.append(1)
            else:
                truth_list.append(0)
        else:
            truth_list.append(2)

    cell_LR_data.y = torch.Tensor(truth_list).type(torch.LongTensor)

    truth_array = np.array(truth_list)
    positive_classes = np.where(truth_array==1)[0].tolist()
    #negative_classes = np.where(truth_array==0)[0].tolist()[:len(positive_classes)]
    negative_classes = np.where(truth_array==0)[0].tolist()
    new_train_mask = np.array([False]*truth_array.shape[0])
    new_train_mask[positive_classes + negative_classes] = True
    data = cell_LR_data.to(device)
    torch.manual_seed(0)
    if reproduce is not None:
        print("Reproducing results from original paper - no training process")
        model = torch.load(f"//data/models/{reproduce}/cell_LR.pt")
    else:
        model = GAT(cell_LR_data,num_classes=2).to(device)
        data.train_mask = torch.Tensor(new_train_mask).type(torch.LongTensor)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        ligands = cell_LR_nodes[cell_LR_nodes["category"]=="Ligand"]["Id"].tolist()
        receptors = cell_LR_nodes[cell_LR_nodes["category"]=="Receptor"]["Id"].tolist()

        truth_df = full_matrix.loc[cell_LR_nodes[cell_LR_nodes["category"]=="Ligand"]["identifier"].tolist(),cell_LR_nodes[cell_LR_nodes["category"]=="Receptor"]["identifier"].tolist()]

        truth_Tensor = torch.Tensor(truth_df.values).to(device)

        for epoch in range(100):

            model.train()
            optimizer.zero_grad()
            out = model(data)
            ligand_out = out[ligands,:]
            receptor_out = out[receptors,:]
            total_out = torch.inner(ligand_out,receptor_out)
            loss = criterion(out[new_train_mask],data.y[new_train_mask])
            if epoch % 20 == 0:
                print(f"Epoch:{epoch} Loss:{loss}")
            loss.backward()

            optimizer.step()
        os.system(f"mkdir -p //data/models/{save}/")
        torch.save(model,f"//data/models/{save}/cell_LR.pt")
    model.eval()
    cell_LR_out = model(data)

    ligands = cell_LR_nodes[cell_LR_nodes["category"]=="Ligand"]["Id"].tolist()
    receptors = cell_LR_nodes[cell_LR_nodes["category"]=="Receptor"]["Id"].tolist()

    ligand_out = cell_LR_out[ligands,:]
    receptor_out = cell_LR_out[receptors,:]
    _,ligand_pred = ligand_out.max(dim=1)
    _,receptor_pred = receptor_out.max(dim=1)

    total_out = torch.inner(ligand_out,receptor_out).cpu().detach().numpy()
    
    cell_LR_nodes.index = cell_LR_nodes["Id"].tolist()


    ligands_df = cell_LR_nodes[cell_LR_nodes["category"]=="Ligand"]
    receptors_df = cell_LR_nodes[cell_LR_nodes["category"]=="Receptor"]
    ligands_df.index = range(ligands_df.shape[0])
    receptors_df.index = range(receptors_df.shape[0])

    valid_ligands = ligands_df
    valid_receptors = receptors_df
    
    ligand_pred = ligand_pred.cpu().detach().numpy()
    receptor_pred = receptor_pred.cpu().detach().numpy()
    
    cell_LR_nodes.index = cell_LR_nodes["Id"].tolist()
    total_out = torch.inner(ligand_out,receptor_out).cpu().detach().numpy()
    ligand_nodes = cell_LR_nodes[cell_LR_nodes["category"] == "Ligand"]
    ligand_nodes.index = range(0,ligand_nodes.shape[0])
    ligand_idents = ligand_nodes.iloc[np.where(ligand_pred==1)]['identifier'].tolist()
    total_out_df = pd.DataFrame(total_out,index=valid_ligands["identifier"].tolist(),columns=valid_receptors["identifier"].tolist())
    indicies = np.where(total_out_df.values > -100)
    source = list(indicies[0])
    dest = list(indicies[1])
    index_df = pd.DataFrame({"Id":range(0,total_out_df.shape[0]),"identifier":total_out_df.index.tolist()})
    column_df = pd.DataFrame({"Id":range(0,total_out_df.shape[1]),"identifier":total_out_df.columns.tolist()})
    source_list = index_df.loc[source]["identifier"].tolist()
    dest_list = column_df.loc[dest]["identifier"].tolist()
    total_link_df = pd.DataFrame({"Src":source_list,"Dst":dest_list,"Prob":total_out_df.values[indicies]})

    total_link_df = total_link_df.sort_values("Prob",ascending=False)
    Omnipath_db = pd.read_csv("//data/LR_database/Omnipath_database.csv",index_col=0)
    total_link_df["Src"] = [i.split("_")[0] for i in total_link_df["Src"].tolist()]
    total_link_df["Dst"] = [i.split("_")[0] for i in total_link_df["Dst"].tolist()]
    total_link_df = total_link_df.drop_duplicates()
    
    if spatial is not None:
        LR_out = cell_LR_out[valid_ligands["Id"].tolist() + valid_receptors["Id"].tolist(),:]
        cell_group_out = cell_LR_out[cell_LR_nodes[cell_LR_nodes["category"]=="Cell Group"]["Id"].tolist(),:]
        new_cell_LR_out = torch.inner(LR_out,cell_group_out).cpu().detach().numpy()

        cell_LR_df = pd.DataFrame(new_cell_LR_out,index=valid_ligands["identifier"].tolist() + valid_receptors["identifier"].tolist(),columns=cell_LR_nodes[cell_LR_nodes["category"]=="Cell Group"]["identifier"].tolist())

        ligands = nodes[nodes["category"]=="Ligand"]["identifier"].tolist()
        receptors = nodes[nodes["category"]=="Receptor"]["identifier"].tolist()
        expression_df = matrix

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

        ligands = cell_LR_nodes[cell_LR_nodes["category"]=="Ligand"]["Id"].tolist()
        receptors = cell_LR_nodes[cell_LR_nodes["category"]=="Receptor"]["Id"].tolist()

        interacting_ligands = list(set(ligands) & set(cell_LR_ints["Dst"].tolist()))
        interacting_receptors = list(set(receptors) & set(cell_LR_ints["Dst"].tolist()))
        cell_LR_ints.index = cell_LR_ints["Dst"].tolist()
        ligand_cells = cell_LR_ints.loc[interacting_ligands]["Src"].unique().tolist()
        receptor_cells = cell_LR_ints.loc[interacting_receptors]["Src"].unique().tolist()

        #cell_LR_out = torch.Tensor(cell_LR_out)

        ligand_cell_out = cell_LR_out[ligand_cells,:]
        ligand_out = cell_LR_out[ligands,:]
        total_ligand_out = torch.inner(ligand_out,ligand_cell_out).cpu().detach().numpy()
        receptor_cell_out = cell_LR_out[receptor_cells,:]
        receptor_out = cell_LR_out[receptors,:]
        total_receptor_out = torch.inner(receptor_out,receptor_cell_out).cpu().detach().numpy()

        ligand_matrix = mean_matrix[mean_matrix.index.str.contains("Ligand")]
        receptor_matrix = mean_matrix[mean_matrix.index.str.contains("Receptor")]
        ligand_cell_out = np.multiply(ligand_matrix,total_ligand_out)
        receptor_cell_out = np.multiply(receptor_matrix,total_receptor_out)
        ligand_maxes = (ligand_cell_out.idxmax(axis=1))
        receptor_maxes = (receptor_cell_out.idxmax(axis=1))


        ligand_maxes.index = [i.split("_")[0] for i in ligand_maxes.index.tolist()]
        receptor_maxes.index = [i.split("_")[0] for i in receptor_maxes.index.tolist()]

        total_link_df["Src Cell"] = ligand_maxes.loc[total_link_df['Src']].tolist()
        total_link_df["Dst Cell"] = receptor_maxes.loc[total_link_df['Dst']].tolist()

    return total_link_df


def make_nodes_interactions(matrix,input_meta=None):
    matrix.index = [str(i).upper() for i in matrix.index.tolist()]



    index = matrix.index.tolist()

    matrix = matrix.fillna(0)

    adata = anndata.AnnData(matrix.transpose())

    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.log1p(adata)
    sc.pp.normalize_total(adata)
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata)
    if input_meta is None:
        meta = pd.DataFrame({"cell":adata.obs["leiden"].index.tolist(),"labels":adata.obs["leiden"].tolist()})
    else:
        meta = input_meta
    meta.index = meta["cell"].tolist()
    adata.obs = meta

    if isinstance(adata.X,scipy.sparse._csr.csr_matrix):
        matrix = pd.DataFrame.sparse.from_spmatrix(adata.X.transpose(),columns=adata.obs.index.tolist(),index=adata.var.index.tolist())
    else:
        matrix = pd.DataFrame(adata.X.transpose(),columns=adata.obs.index.tolist(),index=adata.var.index.tolist())
    

    matrix = matrix.loc[adata.var.index.tolist()]


    # In[ ]:


    #sc.tl.rank_genes_groups(adata, 'labels')
    cell_groups = meta['labels'].unique().tolist()
    matrix_list = {}
    for i in cell_groups:
        cells = meta[meta["labels"]==i].index.tolist()
        temp_matrix = matrix[cells]
        matrix_list[i] = (temp_matrix.mean(axis=1)[temp_matrix.mean(axis=1) > 0].index.tolist())

    cell_type_df = matrix_list


    # In[23]:


    #mean_expression = np.mean(matrix.values[matrix.values != 0])
    nodes = pd.DataFrame({"category":[],"identifier":[]})


    # In[24]:


    LR_nodes = pd.read_csv("//data/LR_database/nodes.csv",index_col=0)
    Omnipath_network = pd.read_csv("//data/LR_database/interactions.csv",index_col=0)


    # In[25]:


    ligands = LR_nodes[LR_nodes["category"]=="Ligand"]["identifier"].tolist()
    receptors = LR_nodes[LR_nodes["category"]=="Receptor"]["identifier"].tolist()

    ligand_list = []
    receptor_list = []
    new_cell_df = {}
    for i in cell_type_df.keys():
        ligand_list.extend(list(set(ligands) & set(cell_type_df[i])))
        receptor_list.extend(list(set(receptors) & set(cell_type_df[i])))
        new_cell_df[i] = [list(set(ligands) & set(cell_type_df[i])),list(set(receptors) & set(cell_type_df[i]))]


    # In[27]:


    for i in new_cell_df.keys():
        new_cell_df[i][0] = [j+"_Ligand" for j in new_cell_df[i][0]]
        new_cell_df[i][1] = [j+"_Receptor" for j in new_cell_df[i][1]]


    # In[28]:


    ligand_list = list(set(ligand_list))
    receptor_list = list(set(receptor_list))


    # In[29]:


    ligand_list = [i+"_Ligand" for i in ligand_list]
    receptor_list = [i+"_Receptor" for i in receptor_list]


    # In[30]:


    nodes = pd.concat([nodes,pd.DataFrame({"category":["Cell Group"] * len(list(cell_type_df.keys())), "identifier":list(cell_type_df.keys())})])
    nodes = pd.concat([nodes,pd.DataFrame({"category":["Ligand"] * len(ligand_list), "identifier":ligand_list})])
    nodes = pd.concat([nodes,pd.DataFrame({"category":["Receptor"] * len(receptor_list), "identifier":receptor_list})])


    # In[31]:


    new_identifier = [row["identifier"] + "_" + row["category"] for index,row in LR_nodes.iterrows()]


    # In[32]:


    LR_nodes["identifier"] = new_identifier


    # In[33]:


    nodes["Id"] = range(0,nodes.shape[0])
    nodes = nodes[["Id","category","identifier"]]


    # In[34]:


    nodes.index = nodes.index.astype('int')
    nodes["Id"] = nodes["Id"].astype('int')


    # In[35]:


    meta.index = meta["cell"]


    # In[36]:


    interactions = pd.DataFrame({"Src":[],"Dst":[],"Weight":[],"edge_type":[]})


    # In[37]:


    LR_nodes.index = LR_nodes["Id"].tolist()


    # In[38]:


    Omnipath_network["Src"] = LR_nodes.loc[Omnipath_network["Src"].tolist()]["identifier"].tolist()
    Omnipath_network["Dst"] = LR_nodes.loc[Omnipath_network["Dst"].tolist()]["identifier"].tolist()


    # In[39]:


    source_list = []
    dest_list = []
    weight_list = []
    edge_type_list = []
    for i in new_cell_df.keys():
        source_list.extend([i] * (len(new_cell_df[i][0]) + len(new_cell_df[i][1])))
        dest_list.extend(new_cell_df[i][0])
        dest_list.extend(new_cell_df[i][1])
        weight_list.extend([1] * (len(new_cell_df[i][0]) + len(new_cell_df[i][1])))
        edge_type_list.extend([1] * (len(new_cell_df[i][0]) + len(new_cell_df[i][1])))


    # In[40]:


    interactions["Src"] = source_list
    interactions["Dst"] = dest_list
    interactions["Weight"] = weight_list
    interactions["edge_type"] = edge_type_list


    # In[41]:


    nodes.index = nodes["identifier"].tolist()


    # In[42]:


    nodes = nodes.drop_duplicates("identifier")
    nodes["Id"] = range(0,nodes.shape[0])


    # In[43]:


    interactions["Src"] = nodes.loc[interactions["Src"].tolist()]["Id"].tolist()
    interactions["Dst"] = nodes.loc[interactions["Dst"].tolist()]["Id"].tolist()

    LR_nodes = pd.read_csv("//data/LR_database/nodes.csv",index_col=0)
    Omnipath_network = pd.read_csv("//data/LR_database/interactions.csv",index_col=0)
    LR_nodes.index = LR_nodes["Id"].tolist()

    # In[55]:


    new_identifier = [row["identifier"] + "_" + row["category"] for index,row in LR_nodes.iterrows()]


    # In[56]:


    LR_nodes["identifier"] = new_identifier


    # In[57]:


    Omnipath_network["Src"] = LR_nodes.loc[Omnipath_network["Src"].tolist()]["identifier"].tolist()
    Omnipath_network["Dst"] = LR_nodes.loc[Omnipath_network["Dst"].tolist()]["identifier"].tolist()


    # In[58]:


    LR_nodes = LR_nodes[(LR_nodes["identifier"].isin(ligand_list)) | (LR_nodes["identifier"].isin(receptor_list))]


    # In[59]:


    Omnipath_network = Omnipath_network[(Omnipath_network["Src"].isin(LR_nodes["identifier"].tolist())) & (Omnipath_network["Dst"].isin(LR_nodes["identifier"].tolist()))]


    # In[60]:


    LR_nodes["Id"] = range(0,LR_nodes.shape[0])


    # In[61]:


    LR_nodes.index = LR_nodes["identifier"].tolist()
    Omnipath_network["Src"] = LR_nodes.loc[Omnipath_network["Src"].tolist()]["Id"].tolist()
    Omnipath_network["Dst"] = LR_nodes.loc[Omnipath_network["Dst"].tolist()]["Id"].tolist()


    # In[62]:


    LR_nodes = LR_nodes[(LR_nodes["Id"].isin(Omnipath_network["Src"].tolist())) | (LR_nodes["Id"].isin(Omnipath_network["Dst"].tolist()))]


    # In[63]:


    LR_nodes.index = LR_nodes["Id"].tolist()
    Omnipath_network["Src"] = LR_nodes.loc[Omnipath_network["Src"].tolist()]["identifier"].tolist()
    Omnipath_network["Dst"] = LR_nodes.loc[Omnipath_network["Dst"].tolist()]["identifier"].tolist()


    # In[64]:


    LR_nodes.index = LR_nodes["identifier"].tolist()
    LR_nodes["Id"] = range(0,LR_nodes.shape[0])
    Omnipath_network["Src"] = LR_nodes.loc[Omnipath_network["Src"].tolist()]["Id"].tolist()
    Omnipath_network["Dst"] = LR_nodes.loc[Omnipath_network["Dst"].tolist()]["Id"].tolist()

    return matrix,meta,nodes,interactions,LR_nodes,Omnipath_network
