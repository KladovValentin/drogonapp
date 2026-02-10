import networkx as nx
import matplotlib.pyplot as plt
import torch

from predict import loadModel
from dataHandling import make_graph
from config import Config

mainPath = "/home/localadmin_jmesschendorp/gsiWorkFiles/realTimeCalibrations/backend/serverData/"
path = mainPath+"function_prediction/"

def get_grid_layout(rows=2, cols=6):
    pos = {}
    for i in range(rows):
        for j in range(cols):
            node_id = i * cols + j
            pos[node_id] = (j, -i)  # Use -i to keep row 0 on top
    return pos

def plot_graph(threshold=None):

    e_i, e_a = make_graph(Config())

    input_dim = torch.Size([1, Config().channelsLength+1, 24])

    nn_model = loadModel(Config(), input_dim, 1, path, e_i=(torch.LongTensor(e_i).movedim(-2,-1)), e_a=torch.Tensor(e_a))
    edge_weights = nn_model.e_a.detach().cpu().numpy()
    edge_index = nn_model.e_i.cpu().numpy()  # shape: (2, num_edges)

    G = nx.DiGraph()  # Use nx.Graph() for undirected

    src_nodes = edge_index[0]
    tgt_nodes = edge_index[1]

    for i in range(len(edge_weights)):
        weight = edge_weights[i]
        if threshold is None or weight > threshold:
            G.add_edge(int(src_nodes[i]), int(tgt_nodes[i]), weight=weight)

    pos = nx.spring_layout(G, seed=42)  # or nx.circular_layout(G)

    edges = G.edges()
    weights = [G[u][v]['weight'] for u,v in edges]

    pos = get_grid_layout(rows=4, cols=6)

    plt.figure(figsize=(8,6))
    #nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700,
    #    edgelist=edges, edge_color=weights, edge_cmap=plt.cm.viridis, width=2)
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=800)
    nx.draw_networkx_labels(G, pos)

    nx.draw_networkx_edges(G, pos, edgelist=edges,
                           edge_color=weights,
                           edge_cmap=plt.cm.viridis,
                           width=2)
                           #connectionstyle='arc3,rad=0.2')

    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                               norm=plt.Normalize(vmin=min(weights), vmax=max(weights)))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label("Edge Weight", fontsize=14)
    plt.title("")
    plt.show()

plot_graph()