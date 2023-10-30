#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.layers import Input, Dense, GaussianNoise
from keras.models import Model
from keras import regularizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import networkx as nx
import pandas as pd
from utils import DataGenerator, tsne
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
import logging
import sys
import pdb
from sklearn.cluster import KMeans
import pickle

def read_graph(filename,ep_name = None):
    # with open(filename,'rb') as f:
    #     if g_type == "undirected":
    #         G = nx.read_weighted_edgelist(f)
    #     else:
    #         G = nx.read_weighted_edgelist(f,create_using=nx.DiGraph())
    G = nx.read_gml(filename)
    node_idx = G.nodes()
    if ep_name == None:
      adj_matrix = np.asarray(nx.adjacency_matrix(G, nodelist=None).todense())
    else:
      adj_matrix = np.asarray(nx.adjacency_matrix(G, nodelist=None,weight='weight').todense())
    return adj_matrix, node_idx

def scale_sim_mat(mat):
	# Scale Matrix by row
	mat  = mat - np.diag(np.diag(mat))
	D_inv = np.diag(np.reciprocal(np.sum(mat,axis=0),dtype='float32'))
	D_inv[np.isinf(D_inv)] = 1
	mat = np.dot(D_inv,  mat)

	return mat

def random_surfing(adj_matrix,max_step,alpha):
	# Random Surfing
	nm_nodes = len(adj_matrix)
	adj_matrix = scale_sim_mat(adj_matrix)
	P0 = np.eye(nm_nodes, dtype='float32')
	M = np.zeros((nm_nodes,nm_nodes),dtype='float32')
	P = np.eye(nm_nodes, dtype='float32')
	for i in range(0,max_step):
		P = alpha*np.dot(P,adj_matrix) + (1-alpha)*P0
		M = M + P

	return M

def PPMI_matrix(M):

	M = scale_sim_mat(M)
	nm_nodes = len(M)
	col_s = np.sum(M, axis=0).reshape(1,nm_nodes)
	row_s = np.sum(M, axis=1).reshape(nm_nodes,1)
	D = np.sum(col_s)
	rowcol_s = np.dot(row_s,col_s)
	PPMI = np.log(np.divide(D*M,rowcol_s))
	PPMI[np.isnan(PPMI)] = 0.0
	PPMI[np.isinf(PPMI)] = 0.0
	PPMI[np.isneginf(PPMI)] = 0.0
	PPMI[PPMI<0] = 0.0

	return PPMI


def model(data, hidden_layers, hidden_neurons, output_file, validation_split=0.9):
    train_n = int(validation_split * len(data))
    batch_size = 50
    train_data = data[:train_n,:]
    val_data = data[train_n:,:]

    input_sh = Input(shape=(data.shape[1],))
    encoded = GaussianNoise(0.2)(input_sh)
    for i in range(hidden_layers):
        encoded = Dense(hidden_neurons[i], activation='relu')(encoded)
        encoded = GaussianNoise(0.2)(encoded)

    decoded = Dense(hidden_neurons[-2], activation='relu')(encoded)
    for j in range(hidden_layers-3,-1,-1):
        decoded = Dense(hidden_neurons[j], activation='relu')(decoded)
    decoded = Dense(data.shape[1], activation='sigmoid')(decoded)

    autoencoder = Model(input=input_sh, output=decoded)
    autoencoder.compile(optimizer='adadelta', loss='mse')

    checkpointer = ModelCheckpoint(filepath= output_file + ".hdf5", verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=15, verbose=1)

    train_generator = DataGenerator(batch_size)
    train_generator.fit(train_data, train_data)
    val_generator = DataGenerator(batch_size)
    val_generator.fit(val_data, val_data)

    autoencoder.fit_generator(train_generator,
		samples_per_epoch=len(train_data),
		nb_epoch=100,
		validation_data=val_generator,
		nb_val_samples=len(val_data),
		max_q_size=batch_size,
		callbacks=[checkpointer, earlystopper])
    enco = Model(input=input_sh, output=encoded)
    enco.compile(optimizer='adadelta', loss='mse')
    reprsn = enco.predict(data)
    return reprsn


def DNGR_partition(g,k = 10,ep_name = None):
    '''
        find community via deep neural networks for graph representations algorithm
        Parameters: networkx network g
            k: int number of spectral_communities
            ep_name: string the edge weight name
        Return: list of partition
    '''
    Ksteps = 10 #Number of steps for random surfing
    alpha = 0.98 #alpha random surfing
    hidden_layers = 3 #AutoEnocoder Layers
    hidden_neurons = [128,64,32] #Number of Neurons AE
    output_file = 'temp_checkpoints'
    node_idx = g.nodes()
    if ep_name == None:
      adj_matrix = np.asarray(nx.adjacency_matrix(g, nodelist=None).todense())
    else:
      adj_matrix = np.asarray(nx.adjacency_matrix(g, nodelist=None,weight='weight').todense())

    data_mat = adj_matrix
    data = random_surfing(data_mat, Ksteps, alpha)
    data = PPMI_matrix(data)
    reprsn = model(data,hidden_layers,hidden_neurons,output_file)
    reprsn = [np.asarray(row,dtype='float32') for row in reprsn]
    reprsn = np.array(reprsn, dtype='float32')
    km = KMeans(init='k-means++', n_clusters=20, n_init=10)
    km.fit(reprsn)

    return list(km.labels_);