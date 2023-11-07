# Classifying community structure types in networks

This code accompanies the paper <i> "Non-assortative relationships between groups of nodes are common in complex networks"</i>, by Cathy Liu, [Tristram J. Alexander](https://www.sydney.edu.au/science/about/our-people/academic-staff/tristram-alexander.html), and [Eduardo G. Altmann](https://www.maths.usyd.edu.au/u/ega/), [PNAS Nexus (2023)](https://doi.org/10.1093/pnasnexus/pgad364). 

The analysis is performed in two steps:

1. Given a network (in folder Data/), use one of the 5 community-detection methods (SBM, Louvain, Spectral, Infomap, or DNGR, see below) to partition the nodes in groups (stored in the folder Output/).

2. Given a network partition (in folder Output/), classify the relationship between groups/communities in one of the four types: Assortative, Disassortative, Core-Periphery, or Source-Basin.

The Tutorial.ipynb notebook shows how to compute the partitions (step 1) and analyze pre-computed partitions of 52 networks and 5 methods (step 2). 

## Community detection methods

| Method | Reference | Requirement
| --- | --- | --- |
| `SBM` | [Bayesian stochastic blockmodeling](https://onlinelibrary.wiley.com/doi/abs/10.1002/9781119483298.ch11)  | [graph-tool](https://graph-tool.skewed.de/) |
| `Louvain` | [Fast unfolding of communities in large networks](https://iopscience.iop.org/article/10.1088/1742-5468/2008/10/P10008/meta?casa_token=yHKOJQwT8pcAAAAA:xOHs5RUTN077PE9DLbbgM0Hlxh76tj9PYfRtCdpEAHsc-ztBhqasGtV-C14QIlkDV7-um55c2g) | [python-louvain](https://python-louvain.readthedocs.io/en/latest/) |
| `Spectral` | [Revisiting the Bethe-Hessian: Improved CommunityDetection in Sparse Heterogeneous Graphs](https://proceedings.neurips.cc/paper/2019/hash/3e6260b81898beacda3d16db379ed329-Abstract.html) | [codes](https://lorenzodallamico.github.io/codes/) |
| `Infomap` | [Maps of random walks on complex networks reveal community structure](https://www.pnas.org/doi/abs/10.1073/pnas.0706851105) | [infomap](https://mapequation.github.io/infomap/python/) |
| `DNGR` | [Deep neural networks for learning graph representations](https://ojs.aaai.org/index.php/AAAI/article/view/10179) | [DNGR-Keras](https://github.com/MdAsifKhan/DNGR-Keras) under python3.6|


## Repository structure:

1. The notebooks show examples of data analysis:

- Tutorial.ipynb: exemplifies the complete data analysis in one network. 
- DataSurvey.ipynb: full data analysis pipeline used in the paper (52 networks), including: i) choose datasets in 5 domains from repository [Netzschleuder](https://networks.skewed.de); ii) classify structure type for each network iii) produce summary figures (Fig. 4 in paper) 4) and analyse two case studies.

2. src: python files used to produce our result.

  - GenModel.py contains 5 community detection methods with each method running in a seperate file:  SBM.py,Louvain, clustering_more.py, Infomap.py and DNGR.py
   
  - summary_stats.py includes the interaction classification functions;
  
  - Robustness.py includes the boostrapping method to comptue the robustness of the classifications;
  
  - Null.py includes the density null model;
  
  - Converter.py and preprocess.py include helper functions to convert networks between networkx and graph_tool
  
  - summary_stats_all.py is used in DataSurvey.ipynb with more summary functions

3. Data: this folder contains the datasets used in our case study, including the 52 cases downloaded from [Netzschleuder](https://networks.skewed.de). Additional datasets can be stored and analyzed from here.

4. Outputs: This folder contains community partition results by 5 methods in 52 networks in 5 domains (Online&Social, Economic, Biological, Technological, Informational).
