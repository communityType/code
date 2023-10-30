import networkx as nx
from community import community_louvain
import numpy as np
import networkx.algorithms.community as nx_comm

def Louvain_partition(g,ep_name = None, iteration = 10):
    """
        find community via louvain algorithm
        Parameters: networkx network g
            ep_name: string the edge weight name
            iteration: int the number of fitting interation
        Retuen: (list of partition,uncertainty)
    """
    G_undi = g.to_undirected()
    G_undi.remove_edges_from(nx.selfloop_edges(G_undi))

    node_order = [v for v in G_undi.nodes()]

    moud = np.NINF
    if ep_name == None:
        for i in range(iteration):
            par_tmp = convert_louvain(node_order,community_louvain.best_partition(G_undi,random_state = i))
            iter_temp = convert_par_to_iter(G_undi,par_tmp)
            moud_tmp = nx_comm.modularity(G_undi, iter_temp)
            if moud_tmp > moud:
                moud = 1.0*moud_tmp
                par = par_tmp.copy()
            return par,0;
    else:
        for i in range(iteration):
            par_tmp = convert_louvain(node_order,community_louvain.best_partition(G_undi,random_state = i,weight = ep_name))
            iter_temp = convert_par_to_iter(G_undi,par_tmp)
            moud_tmp = nx_comm.modularity(G_undi, iter_temp)
            if moud_tmp > moud:
                moud = 1.0*moud_tmp
                par = par_tmp.copy()
            return par,0;
        
def convert_louvain(node_order,part_dict):
    '''
        Converts partition from louvain into list of communities
        Parameters: node_order: list order of nodes
            part_dict: dictionary contain partition info
        Retuen: list of partition
    '''

    partLouvain_list = [0]*len(part_dict)
    for k in part_dict:
        partLouvain_list[node_order.index(k)] = part_dict[k]
    return partLouvain_list


def convert_par_to_iter(G,par):
    """
        convert a partition list to an iterable partition (for computing modularity)
        Parameters: networkx network G
            par: list contains partition information
        Retuen: partition set with node in one group meaning nodes in one partition
    """
    i = 0
    result = {}
    for n in G.nodes():
        if par[i] in result.keys():
            result[par[i]].add(n)
        else:
            result[par[i]] = {n}

        i += 1

    return result.values();