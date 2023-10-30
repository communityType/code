#!/usr/bin/python3.9

#This file provides five community detecion methods that we used in our analysis:
#   SBM, Greedy(not used), Louvain, Infomap, Spectral CLustering,DNGR


import networkx as nx
#import random
#import numpy as np
#import graph_tool.all as gt
#from collections import Counter,defaultdict
#from networkx.algorithms.community import greedy_modularity_communities
#from community import community_louvain
#import networkx.algorithms.community as nx_comm
#from clustering_more import *
#import infomap
#import DNGR
#from sklearn.cluster import KMeans



def detect_community(G, method, nested = False,degree_correction = True,ep_name = None,iteration = 10,force_B = None):
    '''
        Fit community detection with given method on G
        Parameters: G: networkx graph or grah-tool network
            method: string method name. Only four algorithms implemented: SBM/Greedy/Louvain/Infomap/Spectral/DNGR
            nested: boolean if fit nested sbm
            degree_correction: boolean if fit degree-corrected sbm
            ep_name: string the edge weight name
            iteration: int the number of fitting interation 
        Retuen: list of partition
    '''
    if method == 'SBM':
        from SBM import SBM_partition
        G_par,r = SBM_partition(G,nested = nested,degree_correction = degree_correction,ep_name = ep_name,iteration = iteration,force_B = force_B)
        
    elif method == 'Louvain':
        from Louvain import Louvain_partition
        G_par,r = Louvain_partition(G,ep_name = ep_name,iteration = iteration)
    elif method == 'Infomap':
        from Infomap import Infomap_partition
        G_par = Infomap_partition(G)
    elif method == 'Spectral':
        from Spectral import Spectral_partition
        G_par = Spectral_partition(G)
    elif method == 'DNGR':
        from DNGR import DNGR_partition
        G_par = DNGR_partition(G,ep_name = ep_name)
    return G_par;

