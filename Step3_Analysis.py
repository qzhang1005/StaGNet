# Key Driver Gene Identification and Network Visualization
# This script identifies central genes in the network using a combination of graph metrics, and visualizes the top gene subnet.

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

SEED = 42

def preprocess_network(G):
    """Ensure graph stability and connectivity"""
    G.remove_nodes_from(list(nx.isolates(G)))
    for u, v in G.edges():
        G[u][v]['weight'] = abs(G[u][v].get('weight', 1.0))
    for node in G.nodes():
        G.add_edge(node, node, weight=0.1)
    return G

def identify_key_drivers(network, top_n=100):
    """Identify key driver genes based on multiple centrality measures"""
    G = preprocess_network(nx.from_pandas_adjacency(network, create_using=nx.DiGraph))

    try:
        pr = nx.pagerank(G, alpha=0.90, max_iter=5000, tol=1e-6,
                         weight='weight', nstart={n: 1/G.number_of_nodes() for n in G.nodes()})
    except nx.PowerIterationFailedConvergence:
        pr = nx.hits(G)[0]

    metrics = {
        'pagerank': pr,
        'betweenness': nx.betweenness_centrality(G, weight='weight', seed=SEED),
        'eigenvector': nx.eigenvector_centrality(G, weight='weight', tol=1e-6, max_iter=5000)
    }

    def robust_scale(arr):
        vals = np.array(list(arr.values()))
        med = np.nanmedian(vals)
        q75, q25 = np.nanquantile(vals, 0.75), np.nanquantile(vals, 0.25)
        iqr = q75 - q25
        return {k: (v - med) / iqr if iqr else v - med for k, v in arr.items()}

    pr_s = robust_scale(metrics['pagerank'])
    bt_s = robust_scale(metrics['betweenness'])
    ev_s = robust_scale(metrics['eigenvector'])

    combined = {
        gene: 0.4 * pr_s[gene] + 0.3 * bt_s[gene] + 0.3 * ev_s[gene]
        for gene in G.nodes()
    }

    return sorted(combined.items(), key=lambda x: -x[1])[:top_n]

def visualize_network(network, key_genes):
    """Visualize subnetwork of top key driver genes"""
    sub_net = network.loc[key_genes, key_genes]
    G = preprocess_network(nx.from_pandas_adjacency(sub_net))

    pos = nx.spring_layout(G, seed=SEED, k=1.2, iterations=200)
    node_sizes = [v * 3000 for v in nx.pagerank(G, weight='weight').values()]
    edge_widths = [abs(d['weight']) * 2 for _, _, d in G.edges(data=True)]

    plt.figure(figsize=(16, 12))
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_sizes, cmap='plasma')
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.3)
    nx.draw_networkx_labels(G, pos, font_size=8, bbox=dict(facecolor='white', alpha=0.8))

    plt.title("Stable Gene Network", fontsize=14)
    plt.savefig('Stable_network_Neurons.png', dpi=300, bbox_inches='tight')
    plt.close()
