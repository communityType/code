import networkx as nx
import graph_tool.all as gt
from collections import Counter,defaultdict

def convert_nx_to_gt(G, vp_name = None, ep_name = None):
    """
        convert a networkx network to a graph-tool network
        Parameters: networkx network G
            vp_name: string the unique name of vertex
            ep_name: string the edge weight name
        Retuen: graph-tool network
    """
    N = G.number_of_nodes()

    ## create a graph
    g = gt.Graph(directed=nx.is_directed(G))
    ## define node properties
    ## name: id of user
    name = g.vp[vp_name] = g.new_vp("string")
    if ep_name != None:
        ecount = g.ep[ep_name] = g.new_ep("int")

    node_add = defaultdict(lambda: g.add_vertex())

    ## add all nodes first
    for i_d in G.nodes():
        user_name = i_d
        n=node_add[user_name]
        name[n] = user_name

    ## add all users as nodes
    ## add all retweet/reply as links
    for i_d in G.nodes():
        user_name = i_d
        neighbors = G.neighbors(i_d)

        n=node_add[user_name]
        for nei in neighbors:
            n2=node_add[nei]

            if ep_name != None:
                e = g.add_edge(n, n2)
                ecount[e] = G[n][nei][ep_name]
            else:
                g.add_edge(n,n2)

    return g;

def convert_gt_to_nx(G, vp_name = None, ep_name = None):
    """
        convert a graph-tool network to a networkx network
        Parameters: graph-tool network G
            vp_name: string the unique name of vertex
            ep_name: string the edge weight name
        Retuen: networkx network
    """
    if G.is_directed():
        g = nx.DiGraph()
    else:
        g = nx.Graph()
    if vp_name == None:
        print("No vertex property name is given")
        return;
    for e in G.edges():
        s,t = e
        if ep_name != None:
            g.add_edge(G.vp[vp_name][s],G.vp[vp_name][t],weight = G.ep[ep_name][e])
        else:
            g.add_edge(G.vp[vp_name][s],G.vp[vp_name][t])

    return g;