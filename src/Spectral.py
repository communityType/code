from clustering_more import *
import networkx as nx
import numpy as np

n_cycles = 20
percentage = 'No'
n_classes = 'none'

def Spectral_partition(g,force_B = None):
    """
        find community via spectral clustering algorithm
        Parameters: networkx network g
            force_B: int the number of groups
        Retuen: list of partition
    """
    A = nx.adjacency_matrix(g)
    A = A.todense()
    max_n_classes = g.number_of_nodes()
    if force_B != None:
        spectral_communities, _, _, _ = BH(A, n_cycles,max_n_classes, force_B, percentage)
    else:
        spectral_communities, _, _, _ = BH(A, n_cycles,max_n_classes, n_classes, percentage)
    return spectral_communities.tolist();
