
# In[1]:


# Install required packages.
import os
import torch
os.environ['TORCH'] = torch.__version__
print(torch.__version__)
os.system('pip install -q torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html')
os.system('pip install -q torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html')
os.system('pip install -q git+https://github.com/pyg-team/pytorch_geometric.git')

# Helper function for visualization.
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from utils import *


# In[2]:


import argparse
import os.path as osp
import umap 
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


from torch_geometric.data import Data 

from scipy.spatial.distance import squareform, pdist
def new_visualize(output,num_cells=1000):
    nodes = pd.read_csv("/data/nodes.csv")
    nodes = nodes.drop('Unnamed: 0',axis=1)
    nodes["Id"] = nodes["Id"] - 1
    cos_sim = cosine_similarity(output.detach().numpy())
    cells = nodes[nodes["category"]=="Cell Group"]["identifier"].tolist()
    cells = [i.strip() for i in cells]
    elements = nodes["identifier"].tolist()
    sim_matrix = pd.DataFrame(cos_sim,index=elements,columns=elements)
#     cell_index = []
#     for i in sim_matrix.index.tolist():
#         if i in cells:
#             cell_index.append(i)
#     sim_matrix=sim_matrix.loc[cell_index]
#     sim_matrix = sim_matrix[cell_index]
#     new_elements = sim_matrix.index.tolist()
#     sub = sim_matrix.loc[new_elements[:num_cells]][new_elements[:num_cells]]
#     clustering = SpectralClustering(n_clusters=num_clusters).fit(sub)
#     no_of_colors=num_clusters
#     color=["#"+''.join([random.choice('0123456789ABCDEF') for i in range(6)])
#            for j in range(no_of_colors)]
#     cluster_colours = []
#     for i in clustering.labels_:
#         cluster_colours.append(color[i])
#     sim_TSNE = TSNE(perplexity=200).fit_transform(sub)
#     plt.scatter(sim_TSNE[:,0],sim_TSNE[:,1],color=cluster_colours)
    fig, ax = plt.subplots(figsize=(15,15))
    cax = ax.matshow(cos_sim, interpolation='nearest')
    ax.grid(True)
    plt.title('San Francisco Similarity matrix')
#     plt.xticks(range(33), labels, rotation=90);
#     plt.yticks(range(33), labels);
    fig.colorbar(cax, ticks=[0.0,1.0])
    plt.show()
    
def old_visualize(output,num_clusters=10,num_cells=1000):
    nodes = pd.read_csv("/data/nodes.csv")
    nodes = nodes.drop('Unnamed: 0',axis=1)
    nodes["Id"] = nodes["Id"] - 1
    # cos_sim = cosine_similarity(output.detach().numpy())
    cells = nodes[nodes["category"]=="Cell Group"]["identifier"].tolist()
    cells = [i.strip() for i in cells]
    elements = nodes["identifier"].tolist()
    cells_index = [elements.index(j) for j in cells]
    cells_output = output[cells_index,:].detach().numpy()
    no_of_colors= kmeans.n_clusters
    color=["#"+''.join([random.choice('0123456789ABCDEF') for i in range(6)])
           for j in range(no_of_colors)]
    cluster_colours = []
    for i in kmeans.labels_:
        cluster_colours.append(color[i])
    sim_TSNE = umap.UMAP().fit_transform(cells_output)
    plt.scatter(sim_TSNE[:,0],sim_TSNE[:,1],color=cluster_colours)
def new_cell_group_visualize(output,nodes,meta,separator,image_name,save=False,components=2,mets='euclidean'):
#     nodes = nodes.drop('Unnamed: 0',axis=1)
#     nodes["Id"] = nodes["Id"] - 1
    cos_sim = squareform(pdist(output.detach().numpy()))
    cells = nodes[nodes["category"]=="Cell"]["identifier"].tolist()
    cells = [i.strip() for i in cells]
    elements = nodes["identifier"].tolist()
    cells_index = [elements.index(j) for j in cells]
    sim = pd.DataFrame(cos_sim).loc[cells_index][cells_index]
    #only for cell clustering 
    cells = (pd.read_csv(meta,sep=separator))
    cell_groups = cells["labels"].tolist()
    group_colours = {}
    groups = cells['labels'].unique().tolist()
    for i in groups:
        group_colours[i] = "#"+''.join([random.choice('0123456789ABCDEF') for i in range(6)])
    cell_colours = []
    for j in cell_groups:
        cell_colours.append(group_colours[j])
    sim_TSNE = umap.UMAP(n_components=components,metric=mets).fit_transform(sim)
    df = pd.DataFrame(dict(x=sim_TSNE[:,0],y=sim_TSNE[:,1],groups =cell_groups))
    fig, ax = plt.subplots()
    grouped = df.groupby('groups')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=group_colours[key])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if save:
        plt.tight_layout()
        plt.savefig(f"/results/{image_name}.png")
    plt.show()
    
def old_cell_group_visualize(output,nodes):
#     nodes = pd.read_csv("/data/CC_graph_nodes.csv")
#     nodes = nodes.drop('Unnamed: 0',axis=1)
#     nodes["Id"] = nodes["Id"] - 1
    cos_sim = squareform(pdist(output.detach().numpy()))
    cells = nodes[nodes["category"]=="Cell Group"]["identifier"].tolist()
    cells = [i.strip() for i in cells]
    elements = nodes["identifier"].tolist()
    cells_index = [elements.index(j) for j in cells]
    sim = pd.DataFrame(cos_sim).loc[cells_index][cells_index]
    no_of_colors= 12
    color=["#"+''.join([random.choice('0123456789ABCDEF') for i in range(6)])
           for j in range(no_of_colors)]
    cluster_colours = []
    for i in range(12):
        cluster_colours.append(color[i])
    sim_TSNE = umap.UMAP(n_components=2,metric="jaccard").fit_transform(sim)
    fig, ax = plt.subplots()
    for ix in range(12):
        ax.scatter(sim_TSNE[:,0][ix], sim_TSNE[:,1][ix], c = cluster_colours[ix], label = cells[ix])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

def get_pathway_encodings(pathways,first_pathway,nodes,interactions):
    all_ligands = nodes[nodes['category']=="Ligand"]
    all_ligand_ids = all_ligands["Id"].tolist()
    all_ligands = all_ligands["identifier"].tolist()

    all_receptors = nodes[nodes['category']=="Receptor"]
    all_receptor_ids = all_receptors["Id"].tolist()
    all_receptors = all_receptors["identifier"].tolist()

    all_cells = nodes[nodes['category']=="Cell"]
    all_cells_ids = all_cells["Id"].tolist()
    all_cells = all_cells["identifier"].tolist()

    all_groups = nodes[nodes['category']=="Cell Group"]
    all_group_ids = all_groups["Id"].tolist()
    all_groups = all_groups["identifier"].tolist()

    pathway_members = pathways[pathways["pathway"]==first_pathway]["genesymbol"].tolist()
    # #start with ends - ligands and receptors
    involved_ligands = [i for i in all_ligands if i in pathway_members]
    ligand_encodings = [1 if i in involved_ligands else 0 for i in all_ligands]
    involved_receptors = [i for i in all_receptors if i in pathway_members]
    receptor_encodings = [1 if i in involved_receptors else 0 for i in all_receptors]

    temp_df = pd.DataFrame({"Id": all_ligand_ids + all_receptor_ids, "encoding": ligand_encodings + receptor_encodings})

    #now get cell encodings
    combined_genes = temp_df[temp_df["encoding"]==1]["Id"].tolist()
    potential_sources = interactions[interactions["Dst"].isin(combined_genes)]
    source_cells = potential_sources[potential_sources["Src"].isin(all_cells_ids)]["Src"].unique().tolist()
    cell_encodings = [1 if i in source_cells else 0 for i in all_cells_ids]

    potential_groups = interactions[interactions["Dst"].isin(source_cells)]
    source_groups = potential_groups[potential_groups["Src"].isin(all_group_ids)]["Src"].unique().tolist()
    group_encodings = [1 if i in source_groups else 0 for i in all_group_ids]

    new_encoding_df = pd.DataFrame({"encoding": ligand_encodings + receptor_encodings + cell_encodings + group_encodings}, index= all_ligand_ids + all_receptor_ids + all_cells_ids + all_group_ids)
    new_encoding_df['Id'] = new_encoding_df.index.tolist()
    new_encoding_df.sort_values("Id")
    return new_encoding_df["encoding"].tolist()

import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Parameter
from tqdm import tqdm

#from torch_geometric.datasets import RelLinkPred


set
from torch_geometric.nn import GAE, RGCNConv


class RGCNEncoder(torch.nn.Module):
    def __init__(self, num_nodes, hidden_channels, num_relations,x=None):
        super().__init__()
        self.node_emb = Parameter(torch.Tensor(num_nodes, hidden_channels))
        if x is not None:
            self.x=Parameter(torch.Tensor(x))
            print(self.x)
        else:
            self.x=self.node_emb
        self.conv1 = RGCNConv(hidden_channels, hidden_channels, num_relations)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.node_emb)
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, edge_index, edge_type):
        x = self.x
        x = self.conv1(x, edge_index, edge_type).relu_()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index, edge_type)
        return x


class DistMultDecoder(torch.nn.Module):
    def __init__(self, num_relations, hidden_channels,edges_emb):
        super().__init__()
        self.rel_emb = Parameter(torch.Tensor(num_relations, hidden_channels))
        if edges_emb is not None:
            self.edges_emb=Parameter(torch.Tensor(edges_emb))
            print(self.edges_emb)
        else:
            self.edges_emb=self.rel_emb
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.rel_emb)

    def forward(self, z, edge_index, edge_type):
        z_src, z_dst = z[edge_index[0]], z[edge_index[1]]
        rel = self.rel_emb[edge_type]
        return torch.sum(z_src * rel * z_dst, dim=1)


def negative_sampling(edge_index, num_nodes):
    # Sample edges by corrupting either the subject or the object of each edge.
    mask_1 = torch.rand(edge_index.size(1)) < 0.5
    mask_2 = ~mask_1

    neg_edge_index = edge_index.clone()
    neg_edge_index[0, mask_1] = torch.randint(num_nodes, (mask_1.sum(), ))
    neg_edge_index[1, mask_2] = torch.randint(num_nodes, (mask_2.sum(), ))
    return neg_edge_index


def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.edge_index, data.edge_type)

    pos_out = model.decode(z, data.train_edge_index, data.train_edge_type)

    neg_edge_index = negative_sampling(data.train_edge_index, data.num_nodes)
    neg_out = model.decode(z, neg_edge_index, data.train_edge_type)

    out = torch.cat([pos_out, neg_out])
    gt = torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)])
    cross_entropy_loss = F.binary_cross_entropy_with_logits(out, gt)
    reg_loss = z.pow(2).mean() + model.decoder.rel_emb.pow(2).mean()
    loss = cross_entropy_loss + 1e-2 * reg_loss

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()

    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    z = model.encode(data.edge_index, data.edge_type)

    valid_mrr = compute_mrr(z, data.valid_edge_index, data.valid_edge_type)
    test_mrr = compute_mrr(z, data.test_edge_index, data.test_edge_type)

    return valid_mrr, test_mrr


@torch.no_grad()
def compute_rank(ranks):
    # fair ranking prediction as the average
    # of optimistic and pessimistic ranking
    true = ranks[0]
    optimistic = (ranks > true).sum() + 1
    pessimistic = (ranks >= true).sum()
    return (optimistic + pessimistic).float() * 0.5


@torch.no_grad()
def compute_mrr(z, edge_index, edge_type):
    ranks = []
    for i in tqdm(range(edge_type.numel())):
        (src, dst), rel = edge_index[:, i], edge_type[i]

        # Try all nodes as tails, but delete true triplets:
        tail_mask = torch.ones(data.num_nodes, dtype=torch.bool)
        for (heads, tails), types in [
            (data.train_edge_index, data.train_edge_type),
            (data.valid_edge_index, data.valid_edge_type),
            (data.test_edge_index, data.test_edge_type),
        ]:
            tail_mask[tails[(heads == src) & (types == rel)]] = False

        tail = torch.arange(data.num_nodes)[tail_mask]
        tail = torch.cat([torch.tensor([dst]), tail])
        head = torch.full_like(tail, fill_value=src)
        eval_edge_index = torch.stack([head, tail], dim=0)
        eval_edge_type = torch.full_like(tail, fill_value=rel)

        out = model.decode(z, eval_edge_index, eval_edge_type)
        rank = compute_rank(out)
        ranks.append(rank)

        # Try all nodes as heads, but delete true triplets:
        head_mask = torch.ones(data.num_nodes, dtype=torch.bool)
        for (heads, tails), types in [
            (data.train_edge_index, data.train_edge_type),
            (data.valid_edge_index, data.valid_edge_type),
            (data.test_edge_index, data.test_edge_type),
        ]:
            head_mask[heads[(tails == dst) & (types == rel)]] = False

        head = torch.arange(data.num_nodes)[head_mask]
        head = torch.cat([torch.tensor([src]), head])
        tail = torch.full_like(head, fill_value=dst)
        eval_edge_index = torch.stack([head, tail], dim=0)
        eval_edge_type = torch.full_like(head, fill_value=rel)

        out = model.decode(z, eval_edge_index, eval_edge_type)
        rank = compute_rank(out)
        ranks.append(rank)

    return (1. / torch.tensor(ranks, dtype=torch.float)).mean()

from sklearn.model_selection import train_test_split
from torch_geometric.data import Dataset
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

