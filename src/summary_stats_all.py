import graph_tool.all as gt
from collections import Counter,defaultdict
import networkx.algorithms.community as nx_comm
from Converter import *

import numpy as np
import networkx as nx
import pickle
import os
import pandas as pd
import random
from threading import Thread
import threading

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

def create_par_vp(G,par_name,par_dict):
    """
        create graph-tool vertex property which stores partition informaiton
        Parameters:
            G: graph-tool network
            par_name: string name for the vertex property
            par_dict: dictionary with vertex as key and partition of node as value
        Return: partition vertex property
    """
    par_vp = G.vp[par_name] = G.new_vp("int")
    for n in G.vertices():
        par_vp[n] = par_dict[n]
    return par_vp;

def get_par_dict(G,node_order,par,vp_name="title"):
    '''
        Create a dictionary with vertex as key and partition of this vertex as value
        Parameter: G: graph-tool graph
            node_order: list contains the unique node names. the order of node is the node partition order
            par: list of parition
            vp_name: string the name of vertex property which has unique vertex name
        Return: a dictionary with vertex as key and partition of this vertex as value
    '''
    d = {}
    for v in G.vertices():
        node_name = G.vp[vp_name][v]
        d[v] = par[node_order.index(node_name)]
    return d;

def sort_par(par):
    '''
        sort the given partition based on group size
        return dict {par:sorted_par}
    '''
    count = Counter(par)
    sp = [k for k, v in sorted(count.items(), key=lambda item: item[1],reverse = True)]
    result = {sp[i]:i for i in range(len(sp))}
    return result;

def number_of_groups(par,ep_name = None):
    '''
        sort the given partition based on group size
        return list [par_size]
    '''
    N = len(par) #number of nodes
    min_size = 5
    count = Counter(par)
    sp = [v for k, v in sorted(count.items(), key=lambda item: item[1],reverse = True)]
    #record all groups larger than min_size
    conf_index = [i for i in range(len(sp)) if sp[i]<=min_size ]

    return conf_index;

def get_degree_par(G,par_dict,ep_name = None,degree = "total"):
    '''
        Compute the average degree of a gt.graph with partition info stored in par_dict
        Parameters: G: graph-tool network
            par_dict: a dictionary with vertex as key and partition of this vertex as value
            ep_name: string, if the network is weighted, ep_name indicates the ep property map name
            degree: string, "total","in","out"
        Return: ndarray average degree of communities
    '''
    B = max(par_dict.values())+1
    m = np.zeros((B))
    par = par_dict
    count_par = Counter(par.values())
    spar = sort_par(par.values())

    if ep_name == None:
        for v in G.vertices():
            if degree == "in":
                m[spar[par[v]]] += G.get_in_degrees([v])[0]/count_par[par[v]]
            elif degree == "out":
                m[spar[par[v]]] += G.get_out_degrees([v])[0]/count_par[par[v]]
            else:
                m[spar[par[v]]] += G.get_total_degrees([v])[0]/count_par[par[v]]
    else:
        eweight = G.ep[ep_name]
        for v in G.vertices():
            if degree == "in":
                m[spar[par[v]]] += G.get_in_degrees([v],eweight=eweight)[0]/count_par[par[v]]
            elif degree == "out":
                m[spar[par[v]]] += G.get_out_degrees([v],eweight=eweight)[0]/count_par[par[v]]
            else:
                m[spar[par[v]]] += G.get_total_degrees([v],eweight=eweight)[0]/count_par[par[v]]

    #remove all 0 entries
    if B != len(set(par_dict.values())):
        i = B-1
        while i >=0:
            if m[i]==0:
                m = np.delete(m,i,axis = 0)
            i -= 1
    return m;


def get_density(G,par_dict,ep_name = None):
    '''
        Compute the density of a gt.graph with partition info stored in par_dict = {vertex:partition}
        Parameters: G: graph-tool network
            par_dict: a dictionary with vertex as key and partition of this vertex as value
            ep_name: string the name of edge property which contains edge weight
        Retuen: ndarray density matrix
    '''
    B = max(par_dict.values())+1
    m = np.zeros((B,B))
    par = par_dict
    count_par = Counter(par.values())
    spar = sort_par(par.values())

    if ep_name == None:
        for e in G.edges():
            s = e.source()
            t = e.target()
            if G.is_directed():
                if par[s] != par[t]:
                    m[spar[par[s]]][spar[par[t]]] += 1/count_par[par[s]]/count_par[par[t]]
                else:
                    m[spar[par[s]]][spar[par[t]]] += 1/count_par[par[s]]/(count_par[par[t]]-1)
            else:
                if par[s] != par[t]:
                    m[spar[par[s]]][spar[par[t]]] += 1/count_par[par[s]]/count_par[par[t]]
                    m[spar[par[t]]][spar[par[s]]] += 1/count_par[par[s]]/count_par[par[t]]
                else:
                    m[spar[par[s]]][spar[par[t]]] += 1/count_par[par[s]]/(count_par[par[t]]-1)*2
    else:
        for e in G.edges():
            s = e.source()
            t = e.target()
            if G.is_directed():
                if par[s] != par[t]:
                    m[spar[par[s]]][spar[par[t]]] += G.ep[ep_name][e]/count_par[par[s]]/count_par[par[t]]
                else:
                    m[spar[par[s]]][spar[par[t]]] += G.ep[ep_name][e]/count_par[par[s]]/(count_par[par[t]]-1)

            else:
                if par[s] != par[t]:
                    m[spar[par[s]]][spar[par[t]]] += G.ep[ep_name][e]/count_par[par[s]]/count_par[par[t]]
                    m[spar[par[t]]][spar[par[s]]] += G.ep[ep_name][e]/count_par[par[s]]/count_par[par[t]]
                else:
                    m[spar[par[s]]][spar[par[t]]] += G.ep[ep_name][e]/count_par[par[s]]/(count_par[par[t]]-1)*2


    #remove all ineligible entries, i.e. not presenting in partition
    if B != len(set(par_dict.values())):
        i = B-1
        n = B
        while i >=0:
            if i not in spar.values():
                m = np.delete(m,i,axis = 1)
                m = np.delete(m,i,axis = 0)
                n -= 1
            i -= 1
    return m;


def pairwise_interaction_type(w00,w11,w01,w10,directed = False):
    """
        compute the exact interaction type(6 for undirected and 24 for directed network)
        Parameters:  pariwise interation density
            directed: boolean if the network is directed
        Return: pairwise interation type as string
            0     1
          =============
        0 | w00 | w01 |
          =============
        1 | w10 | w11 |
          =============
    """
    if directed:
        if w00>=w11 and w11>=w10 and w10>=w01:
            return "D-A1";
        elif w00>=w11 and w11>=w01 and w01>=w10:
            return "D-A2";
        elif w11>=w00 and w00>=w10 and w10>=w01:
            return "D-A3";
        elif w11>=w00 and w00>=w01 and w01>=w10:
            return "D-A4";
        elif w01>=w10 and w10>=w00 and w00>=w11:
            return "D-D1";
        elif w01>=w10 and w10>=w11 and w11>=w00:
            return "D-D7";
        elif w10>=w01 and w01>=w00 and w00>=w11:
            return "D-D2";
        elif w10>=w01 and w01>=w11 and w11>=w00:
            return "D-D8";
        elif w00>=w10 and w10>=w01 and w01>=w11:
            return "D-D5";
        elif w00>=w10 and w10>=w11 and w11>=w01:
            return "D-D6";
        elif w10>=w00 and w00>=w01 and w01>=w11:
            return "D-D4";
        elif w10>=w00 and w00>=w11 and w11>=w01:
            return "D-D3";
        elif w00>=w01 and w01>=w10 and w10>=w11:
            return "D-C1";
        elif w00>=w01 and w01>=w11 and w11>=w10:
            return "D-C2";
        elif w10>=w11 and w11>=w01 and w01>=w00:
            return "D-C5";
        elif w10>=w11 and w11>=w00 and w00>=w01:
            return "D-C6";
        elif w01>=w00 and w00>=w11 and w11>=w10:
            return "D-C4";
        elif w01>=w00 and w00>=w10 and w10>=w11:
            return "D-C3";
        elif w11>=w01 and w01>=w10 and w10>=w00:
            return "D-D9";
        elif w11>=w01 and w01>=w00 and w00>=w10:
            return "D-D10";
        elif w01>=w11 and w11>=w10 and w10>=w00:
            return "D-D11";
        elif w01>=w11 and w11>=w00 and w00>=w10:
            return "D-D12";
        elif w11>=w10 and w10>=w01 and w01>=w00:
            return "D-C7";
        else:
            return "D-C8";

    else:
        if w00 >= w11 and w11 >= w01:
            return "U-A1";
        elif w11 >= w00 and w00 >= w01:
            return "U-A2";
        elif w01 >= w00 and w00 >= w11:
            return "U-D1";
        elif w01 >= w11 and w11 >= w00:
            return "U-D2"
        elif w00 >= w01 and w01 >= w11:
            return "U-C1"
        else:
            return "U-C2";

def pairwise_interaction_mapping(typ):
    '''
        ignore degree and map 24 structure types to 12 types
        Parameters: typ: original type from full classfication list
        Return: string mapping type
    '''
    if typ in ['D-A1','D-A2','D-C1','D-C2','D-C3','D-C4','D-D1','D-D2','D-D3','D-D4','D-D5','D-D6','U-A1','U-C1','U-D1']:
        return typ;
    elif typ == 'D-A3':
        return 'D-A2'
    elif typ == 'D-A4':
        return 'D-A1'
    elif typ == 'D-C7':
        return 'D-C1'
    elif typ == 'D-C8':
        return 'D-C2'
    elif typ == 'D-C5':
        return 'D-C3'
    elif typ == 'D-C6':
        return 'D-C4'
    elif typ == 'D-D7':
        return 'D-D2'
    elif typ == 'D-D8':
        return 'D-D1'
    elif typ == 'D-D9':
        return 'D-D5'
    elif typ == 'D-D10':
        return 'D-D6'
    elif typ == 'D-D11':
        return 'D-D4'
    elif typ == 'D-D12':
        return 'D-D3'
    elif typ == 'U-A2':
        return 'U-A1'
    elif typ == 'U-C2':
        return 'U-C1'
    elif typ == 'U-D2':
        return 'U-D1'
    else:
        return;


def interaction_matrix(den,deg_list,directed = False, all_cases = False):
    '''
        compute pairwise interation matrix Mrs
        Parameters: den: density of network
            deg_list: average degree of each community of network
            directed: boolean if the network is directed
        Return: NxN matrix M_rs
    '''
    N,_ = den.shape
    M = np.ndarray(shape=(N,N),dtype = object)
    for i in range(N):
        for j in range(N):
            if i != j:
                if deg_list[i]>= deg_list[j]:
                    if all_cases:
                        M[i,j] = pairwise_interaction_type(den[i,i],den[j,j],den[i,j],den[j,i],directed)
                    else:
                        M[i,j] = pairwise_interaction_mapping(pairwise_interaction_type(den[i,i],den[j,j],den[i,j],den[j,i],directed))
                else:
                    if all_cases:
                        M[i,j] = pairwise_interaction_type(den[j,j],den[i,i],den[j,i],den[i,j],directed)
                    else:
                        M[i,j] = pairwise_interaction_mapping(pairwise_interaction_type(den[j,j],den[i,i],den[j,i],den[i,j],directed))
            else:
                M[i,j] = " "
    return M;

def three_ineraction_type(mrs):
    """
        conclude an interaction type mrs with exact type to three types: Assortative/Disassortative/Core-Periphery
        Parameters: mrs: string exact interation type
        Retuen: one of three types of interaction
    """
    if  mrs== "U-A1" or mrs== "U-A2" or mrs== "D-A1" or mrs== "D-A2" or mrs== "D-A3" or mrs== "D-A4":
        return "Assortative";
    elif mrs== "U-D1" or mrs== "U-D2" or mrs in ["D-D1","D-D2","D-D7","D-D8"]:
        return "Disassortative";
    elif mrs== "U-C1" or mrs== "U-C2" or mrs in ['D-C1','D-C2','D-C3','D-C4','D-C5','D-C6','D-C7','D-C8']:
        return "Core-Periphery";
    elif mrs == " ":
        return " ";
    elif mrs in ["D-D3","D-D4","D-D5","D-D6","D-D9","D-D10","D-D11","D-D12"]:
        return "Source-Basin"
    else:
        return "Core-Periphery";

def three_interaction_matrix(exact_Mrs):
    """
        conclude an interaction matrix Mrs with exact type to three types: Assortative/Disassortative/Core-Periphery
        Parameters: exact interation matrix
        Retuen: interaction matrix with three types of interaction
    """
    N,_ = exact_Mrs.shape
    M = np.ndarray(shape=(N,N),dtype = object)
    for i in range(N):
        for j in range(N):
            mij = exact_Mrs[i,j]
            M[i,j] = three_ineraction_type(mij)
    return M;


def interaction_type_prob(den, deg_list,directed = False, classification = "All", all_cases = False):
    """
        compute probability of interaction types occuring in interation matrix M_rs (only count large group)
        Parameters: den: density matrix
            deg_list: average degree of each community of network
        Retuen: a dictionary with interation type as key and fraction of occurance as value
    """
    prob = {}
    M_rs = interaction_matrix(den,deg_list,directed, all_cases = all_cases)
    if classification == "Three":
        M_rs = three_interaction_matrix(M_rs)

    N,_ = M_rs.shape
    unique_elements, counts_elements = np.unique(M_rs, return_counts=True)
    for i in range(len(unique_elements)):
        if unique_elements[i] != " ":
            prob[unique_elements[i]] = counts_elements[i]/(N**2-N)
    return prob;

def count_rt(count,typ):
    if typ in count.keys():
        return count[typ];
    else:
        return 0;


def rewire_graph(G, ep_name = None):
    g = G.copy()
    edge_list = [e for e in G.edges()]
    if ep_name == None:
        edge_weight = [1 for e in G.edges()]
    else:
        edge_weight = [g.ep[ep_name][e] for e in G.edges()]
        
    N = sum(edge_weight)
    
    if ep_name == None:
        rewire_edge = random.choices(edge_list, k = N)
    else:
        rewire_edge = random.choices(edge_list, weight = g.ep[ep_name].get_array().tolist(),k = N)

    count = Counter(rewire_edge)
    #        n = rewire_edge.count(e)
    for e in edge_list:
        if e in count.keys():
            n = count[e]
            if ep_name != None:
                g.ep[ep_name][e] = n
        else:
            g.remove_edge(e)


    return g;

def boostrap(G,par_dict,deg_list,lock,reverse_dir = False,ep_name = None):
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

def certainty(G,par_dict,deg_list,reverse_dir = False,ep_name = None, iteration = 10):
    global M_rs,M_rs_three,C_rs,P
    M_rs = []
    M_rs_three = []

    Den = get_density(G,par_dict,ep_name = ep_name)
    if reverse_dir:
        Den = Den.transpose()

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

    return P;

def summary(g,par,node_order,vp_name = "title",ep_name = None,reverse_dir = False):
    '''
        Compute the statistics of a gt.graph with partitions and print the result
        Parameters: g: graph_tool graph
            par: partition 
            node_order: order of unique node attribute "title" 
            vp_name: string vertex property name
            ep_name: string the name of edge property which contains edge weight
        Return: 
    '''
    G_gt = g
    G_nx = convert_gt_to_nx(G_gt, vp_name=vp_name,ep_name=ep_name)
    
    par_dict = get_par_dict(G_gt,node_order=node_order,par = par,vp_name=vp_name)
    par_vp = create_par_vp(G_gt,"par",par_dict)
    
    state = gt.BlockState(G_gt, par_vp)
    print("The number of communities is ",state.get_nonempty_B())
    print("Description length of the partition is ",state.entropy(degree_dl = False))
    print("Modularity of the partition is ",nx_comm.modularity(G_nx, convert_par_to_iter(G_nx,par)))
    
    
    d = get_density(G_gt,par_dict,ep_name = ep_name)
    deg_list = get_degree_par(G_gt,par_dict)
    conf_index = number_of_groups(par,ep_name = ep_name)
    n = len(set(par))-len(conf_index)
    d = np.delete(d, conf_index, 0)
    d = np.delete(d, conf_index, 1)
    deg_list = np.delete(deg_list, conf_index, None)
    print("The number of eligible community is ",n)
    
    if n > 1:
        if reverse_dir:
            d = d.transpose()
        print("Community structure types fractions are as follows ")
        
        inter = interaction_type_prob(d,deg_list,directed = G_gt.is_directed(),classification ="Three")
        for key in inter.keys():
            print(key, " fraction: ",inter[key])
            
    else:   
        print("Do not have enough communities to classify.")
        
    return;

def stats(df,ftype,fname, vp_name = "title",ep_name = None,reverse_dir = False,has_sbm = True,has_sbm_dc = True,has_louvain = True,has_info = True,has_spec=True,has_dngr=True):
    '''
        Compute the statistics of a gt.graph with partitions
        Parameters: df: pandas dataframe with 42 columns as defined in DataSurvey notebook
            vp_name: string vertex property name
            ep_name: string the name of edge property which contains edge weight
            has_sbm: boolean if sbm partition is included
            has_sbm_dc: boolean if sbm_dc partition is included
            has_louvain: boolean if louvain partition is included
            has_info: boolean if infomap partition is included
            has_spec: boolean if spectral partition is included
            has_dngr: boolean if DNGR partition is included
        Return: pandas dataframe with 42 columns
    '''
    path_data = "../Data/"+ftype
    df = pd.concat([df,pd.DataFrame([fname],columns = ['Name'])],ignore_index=True)
    irow = df.shape[0]-1
    fname_data = fname + ".xml.gz"
    filename = os.path.join(path_data,fname_data)
    G_gt = gt.load_graph(filename)
    G_nx = convert_gt_to_nx(G_gt, vp_name=vp_name,ep_name=ep_name)
    df.at[irow,'vp_name'] = vp_name
    df.at[irow,'ep_name'] = ep_name
    df.at[irow,'Directed'] = G_gt.is_directed()
    df.at[irow,'N'] = G_gt.num_vertices()
    df.at[irow,'E'] = G_gt.num_edges()
    with open("../Outputs/"+ftype+"/"+fname+".par","rb") as f:
            par = pickle.load(f)

    if has_sbm:
        sbm_par = par['SBM']
        sbm_par_dict = get_par_dict(G_gt,node_order=[G_gt.vp[vp_name][n] for n in G_gt.vertices()],par = sbm_par,vp_name=vp_name)
        sbm_par_vp = create_par_vp(G_gt,"sbm",sbm_par_dict)
        sbm_state = gt.BlockState(G_gt, sbm_par_vp)
        df.at[irow,'Community Size-SBM'] = sbm_state.get_nonempty_B()
        df.at[irow,'Description Length-SBM'] = sbm_state.entropy(degree_dl = False)
        df.at[irow,'Modularity-SBM'] = nx_comm.modularity(G_nx, convert_par_to_iter(G_nx,sbm_par))

        d_sbm = get_density(G_gt,sbm_par_dict,ep_name = ep_name)
        deg_list = get_degree_par(G_gt,sbm_par_dict)
        conf_index = number_of_groups(sbm_par,ep_name = ep_name)
        n = len(set(sbm_par))-len(conf_index)
        d_sbm = np.delete(d_sbm, conf_index, 0)
        d_sbm = np.delete(d_sbm, conf_index, 1)
        deg_list = np.delete(deg_list, conf_index, None)
        df.at[irow,'Community Eligible Size-SBM'] = n
        if n > 1:
            if reverse_dir:
                d_sbm = d_sbm.transpose()
            df.at[irow,'Interaction Fraction-SBM'] =  interaction_type_prob(d_sbm,deg_list,directed = G_gt.is_directed(),classification ="Three")
            df.at[irow,'Interaction Fraction All-SBM'] = interaction_type_prob(d_sbm,deg_list,directed = G_gt.is_directed(),classification ="All")
        else:
            df.at[irow,'Interaction Fraction-SBM'] = 0
            df.at[irow,'Interaction Fraction All-SBM'] = 0

    if has_sbm_dc:

        sbm_dc_par = par['SBM_DC']
        sbm_dc_par_dict = get_par_dict(G_gt,node_order=[G_gt.vp[vp_name][n] for n in G_gt.vertices()],par =sbm_dc_par,vp_name=vp_name)

        sbm_dc_par_vp = create_par_vp(G_gt,"sbm_dc",sbm_dc_par_dict)
        sbm_dc_state = gt.BlockState(G_gt, sbm_dc_par_vp)

        d_sbm_dc = get_density(G_gt,sbm_dc_par_dict,ep_name)
        deg_list = get_degree_par(G_gt,sbm_dc_par_dict)
        conf_index = number_of_groups(sbm_dc_par,ep_name = ep_name)
        n = len(set(sbm_dc_par))-len(conf_index)
        d_sbm_dc = np.delete(d_sbm_dc, conf_index, 0)
        d_sbm_dc = np.delete(d_sbm_dc, conf_index, 1)
        deg_list = np.delete(deg_list, conf_index, None)
        df.at[irow,'Community Size-SBM_DC'] = sbm_dc_state.get_nonempty_B()
        df.at[irow,'Community Eligible Size-SBM_DC'] = n
        df.at[irow,'Description Length-SBM_DC'] = sbm_dc_state.entropy()
        df.at[irow,'Modularity-SBM_DC'] = nx_comm.modularity(G_nx, convert_par_to_iter(G_nx,sbm_dc_par))
        if n > 1:
            if reverse_dir:
                d_sbm_dc = d_sbm_dc.transpose()
            df.at[irow,'Interaction Fraction-SBM_DC'] = interaction_type_prob(d_sbm_dc,deg_list,directed = G_gt.is_directed(),classification ="Three")
            df.at[irow,'Interaction Fraction All-SBM_DC'] = interaction_type_prob(d_sbm_dc,deg_list,directed = G_gt.is_directed(),classification ="All")
        else:
            df.at[irow,'Interaction Fraction-SBM_DC'] = 0
            df.at[irow,'Interaction Fraction All-SBM_DC'] = 0

    if has_louvain:
        louvain_par = par['LOUVAIN']
        lou_par_dict = get_par_dict(G_gt,node_order=[v for v in G_nx.nodes()],par =louvain_par,vp_name=vp_name)
        lou_par_vp = create_par_vp(G_gt,"louvian",lou_par_dict)
        lou_state = gt.BlockState(G_gt, lou_par_vp)
        d_lou = get_density(G_gt,lou_par_dict,ep_name)
        deg_list = get_degree_par(G_gt,lou_par_dict)
        conf_index = number_of_groups(louvain_par,ep_name = ep_name)
        n = len(set(louvain_par))-len(conf_index)
        d_lou = np.delete(d_lou, conf_index, 0)
        d_lou = np.delete(d_lou, conf_index, 1)
        deg_list = np.delete(deg_list, conf_index, None)
        df.at[irow,'Community Size-Louvain'] = lou_state.get_nonempty_B()
        df.at[irow,'Community Eligible Size-Louvain'] = n
        df.at[irow,'Description Length-Louvain'] = lou_state.entropy(degree_dl = True)
        df.at[irow,'Modularity-Louvain'] = nx_comm.modularity(G_nx, convert_par_to_iter(G_nx,louvain_par))

        if n > 1:
            if reverse_dir:
                d_lou = d_lou.transpose()

            df.at[irow,'Interaction Fraction-Louvain'] = interaction_type_prob(d_lou,deg_list,directed = G_gt.is_directed(),classification ="Three")
            df.at[irow,'Interaction Fraction All-Louvain'] = interaction_type_prob(d_lou,deg_list,directed = G_gt.is_directed(),classification ="All")
        else:
            df.at[irow,'Interaction Fraction-Louvain'] = 0
            df.at[irow,'Interaction Fraction All-Louvain'] = 0

    if has_info:

        info_par = par['INFOMAP']
        info_par_dict = get_par_dict(G_gt,node_order=[v for v in G_nx.nodes()],par =info_par,vp_name=vp_name)
        info_par_vp = create_par_vp(G_gt,"info",info_par_dict)
        info_state = gt.BlockState(G_gt, info_par_vp)
        d_info = get_density(G_gt,info_par_dict,ep_name)
        deg_list = get_degree_par(G_gt,info_par_dict)
        conf_index = number_of_groups(info_par,ep_name = ep_name)
        n = len(set(info_par))-len(conf_index)
        d_info = np.delete(d_info, conf_index, 0)
        d_info = np.delete(d_info, conf_index, 1)
        deg_list = np.delete(deg_list, conf_index, None)
        df.at[irow,'Community Size-Infomap'] = info_state.get_nonempty_B()
        df.at[irow,'Community Eligible Size-Infomap'] = n
        df.at[irow,'Description Length-Infomap'] = info_state.entropy(degree_dl = True)
        df.at[irow,'Modularity-Infomap'] = nx_comm.modularity(G_nx, convert_par_to_iter(G_nx,info_par))

        if n > 1:
            if reverse_dir:
                d_info = d_info.transpose()

            df.at[irow,'Interaction Fraction-Infomap'] = interaction_type_prob(d_info,deg_list,directed = G_gt.is_directed(),classification ="Three")
            df.at[irow,'Interaction Fraction All-Infomap'] = interaction_type_prob(d_info,deg_list,directed = G_gt.is_directed(),classification ="All")
        else:
            df.at[irow,'Interaction Fraction-Infomap'] = 0
            df.at[irow,'Interaction Fraction All-Infomap'] = 0

    if has_spec:
        spec_par = par['SPECTRAL']
        spec_par_dict = get_par_dict(G_gt,node_order=[v for v in G_nx.nodes()],par =spec_par,vp_name=vp_name)
        spec_par_vp = create_par_vp(G_gt,"spec",spec_par_dict)
        spec_state = gt.BlockState(G_gt, spec_par_vp)
        d_spec = get_density(G_gt,spec_par_dict,ep_name)
        deg_list = get_degree_par(G_gt,spec_par_dict)
        conf_index = number_of_groups(spec_par,ep_name = ep_name)
        n = len(set(spec_par))-len(conf_index)
        d_spec = np.delete(d_spec, conf_index, 0)
        d_spec = np.delete(d_spec, conf_index, 1)
        deg_list = np.delete(deg_list, conf_index, None)
        df.at[irow,'Community Size-Spectral'] = spec_state.get_nonempty_B()
        df.at[irow,'Community Eligible Size-Spectral'] = n
        df.at[irow,'Description Length-Spectral'] = spec_state.entropy(degree_dl = True)
        df.at[irow,'Modularity-Spectral'] = nx_comm.modularity(G_nx, convert_par_to_iter(G_nx,spec_par))

        if n > 1:
            if reverse_dir:
                d_spec = d_spec.transpose()

            df.at[irow,'Interaction Fraction-Spectral'] = interaction_type_prob(d_spec,deg_list,directed = G_gt.is_directed(),classification ="Three")
            df.at[irow,'Interaction Fraction All-Spectral'] = interaction_type_prob(d_spec,deg_list,directed = G_gt.is_directed(),classification ="All")
        else:
            df.at[irow,'Interaction Fraction-Spectral'] = 0
            df.at[irow,'Interaction Fraction All-Spectral'] = 0

    if has_dngr:
        dngr_par = par['DNGR']
        dngr_par_dict = get_par_dict(G_gt,node_order=[v for v in G_nx.nodes()],par =dngr_par,vp_name=vp_name)
        dngr_par_vp = create_par_vp(G_gt,"dngr",dngr_par_dict)
        dngr_state = gt.BlockState(G_gt, dngr_par_vp)
        d_dngr = get_density(G_gt,dngr_par_dict,ep_name)
        deg_list = get_degree_par(G_gt,dngr_par_dict)
        conf_index = number_of_groups(dngr_par,ep_name = ep_name)
        n = len(set(dngr_par))-len(conf_index)
        d_dngr = np.delete(d_dngr, conf_index, 0)
        d_dngr = np.delete(d_dngr, conf_index, 1)
        deg_list = np.delete(deg_list, conf_index, None)
        df.at[irow,'Community Size-DNGR'] = dngr_state.get_nonempty_B()
        df.at[irow,'Community Eligible Size-DNGR'] = n
        df.at[irow,'Description Length-DNGR'] = dngr_state.entropy(degree_dl = True)
        df.at[irow,'Modularity-DNGR'] = nx_comm.modularity(G_nx, convert_par_to_iter(G_nx,dngr_par))

        if n > 1:
            if reverse_dir:
                d_dngr = d_dngr.transpose()

            df.at[irow,'Interaction Fraction-DNGR'] = interaction_type_prob(d_dngr,deg_list,directed = G_gt.is_directed(),classification ="Three")
            df.at[irow,'Interaction Fraction All-DNGR'] = interaction_type_prob(d_dngr,deg_list,directed = G_gt.is_directed(),classification ="All")
        else:
            df.at[irow,'Interaction Fraction-DNGR'] = 0
            df.at[irow,'Interaction Fraction All-DNGR'] = 0

    return df;


def get_proportion(df_value,typ):
    value = df_value
    if type(value) == dict:
        if typ in value.keys():
            return value[typ]
        else:
            return 0;
    else:
        return -1;
    
def summary(df,method,irow):
    #network level
    col_name = 'Interaction Fraction-'+method
    values = []
    N_di = 0
    for i in irow:
        if type(ast.literal_eval(df.at[i,col_name])) == dict:
            values.append(ast.literal_eval(df.at[i,col_name]))#only summarise networks with more than 1 community
        if df.at[i,'Directed']:
            N_di += 1
    N = len(values) #number of networks
    
    nonass_prop = 0
    new_prop = 0
    cp_prop = 0
    dis_prop = 0
    sb_prop = 0
    for value in values:
        if get_proportion(value,'Assortative') < 0.999999 :
            nonass_prop += 1
            new = False
            if get_proportion(value,'Core-Periphery') != 0:
                cp_prop += 1
            if get_proportion(value,'Disassortative') != 0:
                new = True
                dis_prop += 1
            if get_proportion(value,'Source-Basin') != 0:
                new = True
                sb_prop += 1
            if new:
                new_prop += 1
    print("==========================================")
    print(method," Summary:")
    print("Among all ",N," networks, ",method," finds ", np.round(nonass_prop/N*100,2),"% networks have non-assortative communities." )
    print("Among all ",N," networks, ",method," finds ", np.round(new_prop/N*100,2),"% networks have communities with new structures." )
    print()
    if nonass_prop != 0:
        print("For networks with non-assortative community structure, ")
        print("   *",np.round(cp_prop/N*100,2),"% has core-periphery structure.")
        print("   *",np.round(dis_prop/N*100,2),"% has disassortative structure.")
        print("   *",np.round(sb_prop/N_di*100,2),"% has source-basin structure.")
        print()
    #pair level
    common_typs = [max(value, key=value.get) for value in values]
#     print("For each network, the most frequent interactions are ",common_typs)
#     print()
    overall_typ = max(set(common_typs), key = common_typs.count)
    print("Overall, the most common interaction type is ",overall_typ, " with fraction ",  np.round(common_typs.count(overall_typ)/N*100,2), "%")
    print("==========================================")
    return common_typs;


def summary_case(G,par,par_name,node_order,vp_name,ep_name = None,reverse_dir = False,fig_size = (16,12)):
    """
        Summarise community structure for a given partition of a network 
        and print out important information including: 
            Reliable community number
            Community structure prevalence
            Density matrix plot with structure type classified and certainty score added
        Parameters:G: graph_tool network
        par: (list) partition information 
        par_name: (string) partition name used for generating vertex property
        node_order: (list) node name in the order of node iterator function of a given package(graph_tool or networkx)
        vp_name: (string) the name of vertex property that stores node name
        ep_name: (string) the name of edge property that stored edge weight (if any)
        reverse_dir: (boolean) if the edge direction is the same as information flow direction
        fig_size: (tuple) the fig size for density matrix plot
        Return: density matrix (transformed if reverse_dir = True)
    """
    par_dict = SS.get_par_dict(G,node_order=node_order,par =par,vp_name=vp_name)
    par_vp = SS.create_par_vp(G,par_name,par_dict)
    state = gt.BlockState(G, par_vp)
    d = SS.get_density(G,par_dict,ep_name)
    deg_list = SS.get_degree_par(G,par_dict)
    m = SS.certainty(G,par_dict,deg_list,reverse_dir = reverse_dir,iteration = 100)['Interation-three']
    conf_index = SS.number_of_groups(par,ep_name = ep_name)
    n = len(set(par))-len(conf_index)
    print("Reliable community number: ",n)
    d = np.delete(d, conf_index, 0)
    d = np.delete(d, conf_index, 1)
    if reverse_dir:
        d = d.transpose()
    deg_list = np.delete(deg_list, conf_index, None)
    p = SS.interaction_type_prob(d, deg_list,directed = G.is_directed(),classification = "Three")
    print(par_name, " has prevalence:",p)
    
    plt.figure(figsize=fig_size)
    annot = SS.interaction_matrix(d,deg_list,directed = G.is_directed())

    N,_ = annot.shape
    for i in range(N):
        for j in range(N):
            if annot[i,j] in ['D-A1','D-A2','D-A3','D-A4','U-A1','U-A2']:
                annot[i,j] = 'A'+'/'+str(round(m[i,j],2))
            elif annot[i,j] in ['D-C1','D-C2','D-C3','D-C4','D-C5','D-C6','D-C7','D-C8','U-C1','U-C2']:
                annot[i,j] = 'CP'+ '/' + str(round(m[i,j],2))
            elif annot[i,j] in ["D-D1","D-D2","D-D7","D-D8","U-D1","U-D2"]:
                annot[i,j] = 'D'+ '/' + str(round(m[i,j],2))
            elif annot[i,j] in ["D-D3","D-D4","D-D5","D-D6","D-D9","D-D10","D-D11","D-D12"]:
                annot[i,j] = 'SB'+ '/' + str(round(m[i,j],2))
            else:
                annot[i,j] = ''
    ax = sns.heatmap(d,annot=annot,fmt = '', cmap="YlGnBu")
    return d;
