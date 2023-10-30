import networkx as nx

def create_vp_name(g,vp_name):
    title = g.vp[vp_name] = g.new_vp("string")
    i = 0
    for v in g.vertices():
        title[v] = str(i)
        i += 1

    return;

def preprocess(G,graph_type = "nx"):
    if graph_type == 'nx':
        import networkx
        if G.is_directed():
            G_undi = G.to_undirected()
            largest = max(nx.connected_components(G_undi), key=len)
            G = nx.DiGraph(G.subgraph(largest))
        else:
            G = G.subgraph(max(nx.connected_components(G_undi), key=len))
            G = nx.Graph(G)
        
        G.remove_edges_from(nx.selfloop_edges(G))
    elif graph_type == 'gt':
        import graph_tool.all as gt
        create_vp_name(G,"title")
        G = gt.extract_largest_component(G, directed = False, prune = True)
        gt.remove_self_loops(G)
    return G;
