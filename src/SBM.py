import networkx as nx
import graph_tool.all as gt
from collections import Counter,defaultdict
import numpy as np


def SBM_partition(g, nested = False, degree_correction = True, ep_name = None, iteration = 10, SM=True,force_B = None):
    """
        use stochastic block model to find community
        Parameters: graph-tool networkx g
            nested: boolean if fit nested sbm
            degree_correction: boolean if fit degree-corrected sbm
            ep_name: string the edge weight name
            iteration: int the number of fitting interation
            force_B: int the number of groups
        Retuen: (list of partition,uncertainty)
    """
    state_arg = {}
    multilevel_mcmc_arg = {}
    if ep_name != None:
        state_arg["eweight"] = g.ep[ep_name]
    if force_B != None:
        multilevel_mcmc_arg['B_min'] = force_B
        multilevel_mcmc_arg['B_max'] = force_B
    state_arg['deg_corr'] = degree_correction

    if SM:
        mdl = np.inf
        if nested:
            for i in range(iteration):
                gt.seed_rng(i)
                state_tmp = gt.minimize_nested_blockmodel_dl(g, state_args=state_arg,multilevel_mcmc_args = multilevel_mcmc_arg)
                mdl_tmp = state_tmp.entropy()
                if mdl_tmp < mdl:
                    mdl = 1.0*mdl_tmp
                    state = state_tmp.copy()
            return list(state.get_bs()),0;
        else:
            for i in range(iteration):
                gt.seed_rng(i)
                state_tmp = gt.minimize_blockmodel_dl(g, state_args=state_arg,multilevel_mcmc_args = multilevel_mcmc_arg)
                mdl_tmp = state_tmp.entropy()
                if mdl_tmp < mdl:
                    mdl = 1.0*mdl_tmp
                    state = state_tmp.copy()
            return list(state.get_blocks().get_array()),0;
    else:
        bs_dg = []
        if nested:
            for i in range(iteration):
                gt.seed_rng(i)
                state = gt.minimize_nested_blockmodel_dl(g, state_args=state_arg,multilevel_mcmc_args = multilevel_mcmc_arg)
                bs_dg.append(list(state.get_bs()))
            # a partition with a maximal overlap to all items of the list of partitions given
            if iteration == 1:
                return G_par,0;
            else:
                G_par, r = gt.nested_partition_overlap_center(bs_dg)
                return G_par, r;
        else:
            for i in range(iteration):
                gt.seed_rng(i)
                state = gt.minimize_blockmodel_dl(g, state_args=state_arg,multilevel_mcmc_args = multilevel_mcmc_arg)
                bs_dg.append(list(state.get_blocks().get_array()))
            # a partition with a maximal overlap to all items of the list of partitions given
            if iteration == 1:
                return G_par,0;
            else:
                G_par, r = gt.partition_overlap_center(bs_dg)
                return G_par, r;