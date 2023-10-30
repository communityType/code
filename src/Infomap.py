import infomap
import networkx as nx

def Infomap_partition(g):
    """
        find community via spectral clustering algorithm
        Parameters: networkx network g
        Retuen: list of partition
    """
    
    im = infomap.Infomap("--two-level")
    im.add_networkx_graph(g)
    im.run()
    communities = im.get_modules()
    return list(communities.values());