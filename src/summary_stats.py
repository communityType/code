from collections import Counter,defaultdict
import GenModel as GM
import numpy as np
import networkx as nx


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

def get_par_dict(G,node_order,par):
    '''
        Create a dictionary with vertex as key and partition of this vertex as value
        Parameter: G: networkx graph
            node_order: list contains the unique node names. the order of node is the node partition order
            par: list of parition
        Return: a dictionary with vertex as key and partition of this vertex as value
    '''
    d = {}
    for v in G.nodes():
        node_name = v
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

def get_degree_par(G,par_dict,ep_name = None):
    '''
        Compute the average degree of a graph with partition info stored in par_dict
        Parameters: G: networkx graph
            par_dict: a dictionary with vertex as key and partition of this vertex as value
            ep_name: string, if the network is weighted, ep_name indicates the ep property map name
        Return: ndarray average degree of communities
    '''
    B = max(par_dict.values())+1
    m = np.zeros((B))
    par = par_dict
    count_par = Counter(par.values())
    spar = sort_par(par.values())

    if ep_name == None:
        for v in G.nodes():
            m[spar[par[v]]] += G.degree(v)/count_par[par[v]]
    else:
        for v in G.nodes():
            m[spar[par[v]]] += G.get_total_degrees(v,weight=ep_name)/count_par[par[v]]

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
        Compute the density of a graph with partition info stored in par_dict = {node:partition}
        Parameters: G: networkx graph
            par_dict: a dictionary with node as key and partition of this node as value
            ep_name: string the name of edge attribute which contains edge weight
        Retuen: ndarray density matrix
    '''
    B = max(par_dict.values())+1
    m = np.zeros((B,B))
    par = par_dict
    count_par = Counter(par.values())
    spar = sort_par(par.values())

    if ep_name == None:
        for e in G.edges():
            s,t = e 
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
            s,t = e 
            w = G[s][t][ep_name]
            if G.is_directed():
                if par[s] != par[t]:
                    m[spar[par[s]]][spar[par[t]]] += w/count_par[par[s]]/count_par[par[t]]
                else:
                    m[spar[par[s]]][spar[par[t]]] += w/count_par[par[s]]/(count_par[par[t]]-1)

            else:
                if par[s] != par[t]:
                    m[spar[par[s]]][spar[par[t]]] += w/count_par[par[s]]/count_par[par[t]]
                    m[spar[par[t]]][spar[par[s]]] += w/count_par[par[s]]/count_par[par[t]]
                else:
                    m[spar[par[s]]][spar[par[t]]] += w/count_par[par[s]]/(count_par[par[t]]-1)*2


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




def summary(g,par,node_order,ep_name = None,reverse_dir = False):
    '''
        Compute the statistics of a graph with partitions and print the result
        Parameters: g: networkx graph
            par: partition 
            node_order: order of unique node attribute, for networkx graph, it is order of node id
            ep_name: string the name of edge attribute which contains edge weight, default: None
            reverse_dir: boolean if the direction of edge needs to be reversed to ally with influence direction, default: false
        Return: None
    '''

    G_nx = g
    
    par_dict = get_par_dict(G_nx,node_order=node_order,par = par)
    
    print("The number of communities is ",len(set(par)))
    
    d = get_density(G_nx,par_dict,ep_name = ep_name)
    deg_list = get_degree_par(G_nx,par_dict)
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
        
        inter = interaction_type_prob(d,deg_list,directed = G_nx.is_directed(),classification ="Three")
        for key in inter.keys():
            print(key, " fraction: ",inter[key])
        M_rs = interaction_matrix(d,deg_list,directed = G_nx.is_directed())
        M_rs = three_interaction_matrix(M_rs)
        return d,M_rs;   
        
    else:   
        print("Do not have enough communities to classify.")
        return;
    return;
            
