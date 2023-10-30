from threading import Thread
import threading
from summary_stats import *
from collections import Counter,defaultdict
import GenModel as GM
import numpy as np
import networkx as nx
import random

def rewire_graph(G, ep_name = None):
    """
        Rewire a graph by randomly picking edges with replacement
    """
    g = G.copy()
    edge_list = [e for e in G.edges()]
    if ep_name == None:
        edge_weight = [1 for (s,t) in G.edges()]
    else:
        edge_weight = [G[s][t]['weight'] for (s,t) in G.edges()]
        
    N = sum(edge_weight)
    
    rewire_edge = random.choices(edge_list, weights = edge_weight,k = N)

    count = Counter(rewire_edge)
    #        n = rewire_edge.count(e)
    for e in edge_list:
        if e in count.keys():
            n = count[e]
            if ep_name != None:
                s,t = e
                g[s][t][ep_name] = n
        else:
            g.remove_edge(*e)

    return g;

def boostrap(G,par_dict,deg_list,lock,reverse_dir = False,ep_name = None):
    """
        Perform boostrapping on a given graph
    """
    global M_rs,M_rs_three
    ret = rewire_graph(G, ep_name = ep_name)
    den = get_density(ret,par_dict,ep_name = ep_name)
    if reverse_dir:
        den = den.transpose()
    M = interaction_matrix(den,deg_list,directed = G.is_directed())
    M_three = three_interaction_matrix(M)


    lock.acquire()
    M_rs.append(M)
    M_rs_three.append(M_three)
    lock.release()

def certainty(G,par,node_order,reverse_dir = False,ep_name = None, iteration = 10):
    """
        Compute the robustness of a given partition
        Parameters: G: networkx graph
            par: partition 
            node_order: order of unique node attribute, for networkx graph, it is order of node id
            ep_name: string the name of edge attribute which contains edge weight, default: None
            reverse_dir: boolean if the direction of edge needs to be reversed to ally with influence direction, default: false
            iteration: the number of boostrapings, default: 10
        Retuen: P: a dictionary with two keys (Interation-all and Interation-three). The correspnding values NxN matrices are the fraction of bootstrapping iterations that classified as same community structure types.
            Interation-all: community structure are classified into 12 configurations
            Interation-three: community structure are classified into 3 (or 4) types (if directed)
    """
    
    global M_rs,M_rs_three,P
    M_rs = []
    M_rs_three = []
    
    par_dict = get_par_dict(G,node_order=node_order,par = par)
    Den = get_density(G,par_dict,ep_name = ep_name)
    deg_list = get_degree_par(G,par_dict)
    if reverse_dir:
        Den = Den.transpose()
    
    conf_index = number_of_groups(par,ep_name = ep_name)
#    print(conf_index)
    n = len(set(par))-len(conf_index)
    
    Mrs = interaction_matrix(Den,deg_list,directed = G.is_directed())
    n,_ = Mrs.shape
    Mrs_three = three_interaction_matrix(Mrs)
    P = {"Interation-all":np.ndarray(shape=(n,n)),"Interation-three":np.ndarray(shape=(n,n))}
    lock = threading.Lock()
    thread_list = []

    for i in range(iteration):
        if ep_name==None:
            thread_list.append(Thread(target=boostrap,args = (G,par_dict,deg_list,lock,reverse_dir)))
        else:
            thread_list.append(Thread(target=boostrap,args = (G,par_dict,deg_list,lock, reverse_dir,ep_name)))
    for thread in thread_list:
        thread.start()
    for thread in thread_list:
        thread.join()

    #calculate proportion

    for r in range(n):
        for c in range(n):
            m = [M_rs[i][r,c] for i in range(iteration)]
            P["Interation-all"][r,c] = m.count(Mrs[r,c])*1.0/iteration
            m = [M_rs_three[i][r,c] for i in range(iteration)]
            P["Interation-three"][r,c] = m.count(Mrs_three[r,c])*1.0/iteration
    
    #remove small communities
    P["Interation-all"] = np.delete(P["Interation-all"], conf_index, 0)
    P["Interation-all"] = np.delete(P["Interation-all"], conf_index, 1)
    
    P["Interation-three"] = np.delete(P["Interation-three"], conf_index, 0)
    P["Interation-three"] = np.delete(P["Interation-three"], conf_index, 1)
    
    return P;