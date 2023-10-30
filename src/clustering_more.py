import numpy as np
import warnings
warnings.filterwarnings("ignore")
import networkx as nx
import numpy as np
import random
random.seed;
import matplotlib.pyplot as plt
import scipy
from scipy import sparse
from scipy.sparse import linalg
import time
import math
from scipy import stats
import sys 
from sklearn.cluster import KMeans


def adj(C_matrix,theta,fractions):

    ''' This function creates the adjacency matrix for more then two classes, according to the DC-SBM model 

    Use: A = adj(C_matrix,theta,fractions)
    Inputs: C_matrix (matrix), theta (vector theta) are the parameters of the DC-SBM model. Fractions is the vector \pi
    Outputs: The adjacency matrix A

    '''
            
    families = len(C_matrix)
    n = len(theta)
    size = (n*fractions).astype(int)
    c_matrix = np.zeros((n,n))
    for i in range(families):
        for j in range(families):
            c_matrix[np.sum(size[:i]):np.sum(size[:i+1])][:,np.sum(size[:j]):np.sum(size[:j+1])] = C_matrix[i][j]

    c_matrix = c_matrix - np.diag(np.diag(c_matrix)) # remove the diagonal term
    c_matrix = np.multiply(theta*np.ones((n,n))*theta[:, np.newaxis],c_matrix) # introduce the theta_itheta_j dependence


    A = np.random.binomial(1,c_matrix/n) # create the matrix  A


    i_lower = np.tril_indices(n, -1)
    A[i_lower] = A.T[i_lower]  # make the matrix symmetric

     
    # connect the unconncted components	

    d = np.sum(A,axis = 0)
    for i in range(n):
        if d[i] == 0:
            first_node = i
            second_node = np.random.choice(np.arange(n), p=theta*c_matrix[i]/sum(theta*c_matrix[i]))
            A[first_node][second_node] = 1
            A[second_node][first_node] = 1
        
    return A



def modularity(A, classes):

    ''' This function computes the modularity of a given class assignment on a network

    Use: mod = modularity(A, classes)
    Inputs: A (adjacency matrix), classes (vector with the label assignment)
    Outputs: The modularity

    '''
    
    n = len(A)
    d = np.sum(A, axis = 0)
    edges = np.sum(d)
    value = 0
    
    for i in range(n):
        for j in range(n):
            if classes[i] == classes[j]:
                value += A[i][j] - d[i]*d[j]/edges
                
    return value/edges   


def L(A,n_clusters):

    ''' This function performs spectral clustering on the informative eigenvectors of the random walk laplacian matrix D^{-1}A for more then two communities.

    Use: y_kmeans, X = L(A,n_clusters)
    Inputs: A a (symmetric) n x n adjacency sparse matrix of a graph, n_clusters (scalar, number of communities in the graph)
    Outputs: y_kmeans the vector of size n of detected classes, X the eigenvector of size n to use for the classification.

    '''
        
    A = A.A
    d = np.sum(A, axis = 0)
    D_1 = np.diag(d**(-1))
    X = np.ones(len(A))
    L = np.dot(D_1,A) # define the matrix
    eigenvalues, eigenvectors = scipy.sparse.linalg.eigs(L, n_clusters, which='LR') # clustering on the largest eigenvectors
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real
    idx = eigenvalues.argsort()[::-1] # sort the eigenvalues
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    kmeans = KMeans(n_clusters = n_clusters) # perform kmeans on the informative eigenvector
    X = np.column_stack((X,eigenvectors[:,1:]))
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    
    return y_kmeans,X


def adjacency(A,n_clusters):

    ''' This function performs spectral clustering on the informative eigenvectors of the adjacency matrix A for more then two communities.

    Use: y_kmeans, X = adjacency(A,n_clusters)
    Inputs: A a (symmetric) n x n adjacency sparse matrix of a graph, n_clusters (scalar, number of communities in the graph)
    Outputs: y_kmeans the vector of size n of detected classes, X the eigenvector of size n to use for the classification.

    '''    
    eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(A, n_clusters, which='LA')
    X = np.zeros(len(A.A))
    idx = eigenvalues.argsort()[::-1] # sort the eigenvalues
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    kmeans = KMeans(n_clusters = n_clusters) # perform kmeans on the informative eigenvector
    X = np.column_stack((X,eigenvectors[:,1:]))
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    
    return y_kmeans,X



def Hessian(A,d,r,n_eig):

    '''This matrix gives the n_eig smallest eigenvalues and the corresponding eigenvectors of the Bethe-Hessian matrix for a given parameter r.'

    Use: eigenvalues, eigenvectors = Hessian(A,d,r,n_eig)
        Inputs: A a (symmetric) n x n adjacency sparse matrix of a graph, d the degree vector, r the value of the parameter of H_r, n_eig number of smallest eienvalues to consider
        Outputs: eigenvalues, eigenvectors (corresponding to the n_eig first smallest)

    '''
    n = len(A.A)
    H = (r**2-1)*np.diag(np.ones(n)) + np.diag(d) - r*A
    eigenvalues,eigenvectors = scipy.sparse.linalg.eigsh(H, n_eig, which='SA')
    idx = eigenvalues.argsort()[::-1] 
    idx = idx[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    
    return eigenvalues, eigenvectors


def overlap(real_classes, classes):

    '''Computes the overlap in neworks with more then two classes and find the good permutaiton of the labels

    Use : classes, ov = overlap(real_classees, classees)
    Input : real_classes (vector of the labels), classes (vector of the estimated labels)
    Output : classes (vector with the good permutation of the classes), ov (overlap)

    '''
    values = max(len(np.unique(real_classes)),len(np.unique(classes))) # number of classes
    n = len(classes) # size of the network

    matrix = np.zeros((values,values))
    for i in range(n):
        matrix[classes[i]][real_classes[i]] += 1 # n_classes x n_classes confusion matrix. Each entry corresponds to how many time label i and label j appeared assigned to the same node

    positions = np.zeros(values)
    for i in range(values):
        positions[i] = np.argmax(matrix[i]) # find the good assignment

    dummy_classes = (classes+1)*100
    for i in range(values):
        classes[dummy_classes == (i+1)*100] = positions[i]

    n_classes = len(np.unique(real_classes))

    ov = (np.sum(classes == real_classes)/n-1/n_classes)/(1-1/n_classes) # compute the overlap

    return classes, ov


def BH(A,n_cycles,max_n_classes,n_classes, percentage):

    ''' This function performs implements our algorithm for the Bethe-Hessian.

        Use: y_kmeans, X, n_clusters, r_estimate = def_L(A,n_cycles,max_n_classes,n_classes, percentage)
        Inputs: A a (symmetric) n x n adjacency sparse matrix of a graph, n_cycles number if iterations for the line search, max_n_classes maximal number of classes admitted, n_classes number of classes of the network; if not kwown set 'none' and it will be estimated, percentage shows the progression of the algo set to 'yes' to see it.
        Outputs: y_kmeans the vector of size n of detected classes, X the eigenvector of size n to use for the classification, n_clusters number of clusters detected, r_estimate vector of the estimated positions of the \zeta

    '''  

    d = np.sum(A.A, axis = 0)
    r_max = np.sqrt(np.sum(d**2)/np.sum(d)) # sqrt(c\Phi)
    r = r_max

    if n_classes == 'none':
        eigenvalues, eigenvectors = Hessian(A,d,r,max_n_classes)
        n_clusters = sum((eigenvalues < 0)*1) # estimate the number of classes
    else:
        n_clusters = n_classes

    r_edge_min = np.ones(n_clusters-1)  # vector containing the lower bound of the different zeta
    r_edge_max = np.ones(n_clusters-1)*r_max  # vector containing the upper bound of the different zeta
    counter = np.zeros(n_clusters-1)
    counter = counter.astype(int)
    n = len(A.A)
    X = np.ones(n)

    for i in range(n_clusters-1):
        init = counter[i]
        for k in range(init,n_cycles): 
            r = 0.5*(r_edge_min[i]+r_edge_max[i]) # iteration of line search
            eigenvalues, eigenvectors = Hessian(A,d,r,n_clusters)
            eigenvalues = eigenvalues[1:]
            r_edge_max[eigenvalues < 0] = np.minimum(r,r_edge_max[eigenvalues < 0]) # updating the upper bound
            r_edge_min[eigenvalues > 0] = np.maximum(r,r_edge_min[eigenvalues > 0]) # updaating the lower bound
            counter[np.where(r == r_edge_max)] += 1 # save one iteration if the bound was updated
            counter[np.where(r == r_edge_min)] += 1 
            counter = np.minimum(counter, n_cycles)
            
            if percentage == 'yes':
            
                OUT = 'Completion : ' + str(np.round(sum(counter)/((n_clusters-1)*n_cycles)*100,1)) +'%'
                sys.stdout.write('\r%s' % OUT)
                sys.stdout.flush()

        X = np.column_stack((X,eigenvectors[:,i+1]))
        
    if X.shape == (n,):
        X = np.column_stack((X,np.ones(n)))

    kmeans = KMeans(n_clusters = n_clusters)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X) 

    return y_kmeans, X, n_clusters, (r_edge_min+r_edge_max)/2


def edge_list(A):

    '''This function returns the edge list representation of a matrix, starting from its adjacency matrix'

    Use : edge_list = edge_list(A)
    Input : A a (symmetric) n x n adjacency sparse matrix of a graph
    Ouput : y.T directed edge list vector


    '''  
    n = len(A)
    edges = np.sum(np.sum(A, axis = 0))
    x = np.where(A == 1)
    y = np.array(x)
    return y.T


def simmetrize(edge_list):

    '''This function makes an unsymmetrix edge list become symmetric

    Use : el = simmetrize(edge_list)
    Input = edge_list non symmetric edge list, el symmetric edge list

    '''
    
    el = np.zeros((2*len(edge_list),2))
    el[:len(edge_list)] = edge_list
    el[len(edge_list):][:,0] = edge_list[:,1]
    el[len(edge_list):][:,1] = edge_list[:,0]
    return el 


def BH_with_r(A,n_cycles,max_n_classes,n_classes, percentage,r_vector):

    '''This function implements our algorithm assuming the optimal positions of r are known

    Use : y_kmeans, X, n_clusters = def_L_with_r(A,n_cycles,max_n_classes,n_classes, percentage,r_vector)
        Inputs: A a (symmetric) n x n adjacency sparse matrix of a graph, n_cycles number if iterations for the line search, max_n_classes maximal number of classes admitted, n_classes number of classes of the network; if not kwown set 'none' and it will be estimated, percentage shows the progression of the algo set to 'yes' to see it, r_vector position of the optimal values of r
        Outputs: y_kmeans the vector of size n of detected classes, X the eigenvector of size n to use for the classification, n_clusters number of clusters detected.

    '''

    d = np.sum(A.A, axis = 0)
    r = np.sqrt(np.sum(d**2)/np.sum(d))

    if n_classes == 'none':
        eigenvalues, eigenvectors = Hessian(A,d,r,max_n_classes)
        n_clusters = sum((eigenvalues < 0)*1) # estimate the number of classes
    else:
        n_clusters = n_classes

    n = len(A.A)
    X = np.ones(n)

    for i in range(n_clusters-1):
        eigenvalues, eigenvectors = Hessian(A,d,r_vector[i],i+2)
        X = np.column_stack((X,eigenvectors[:,-1])) # iteratively find the informative eigenvector
        
    if X.shape == (n,):
        X = np.column_stack((X,np.ones(n)))

    kmeans = KMeans(n_clusters = n_clusters)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X) 

    return y_kmeans, X, n_clusters


def saade(A, n_clusters):

    '''This function implements the Bethe-Hessian as proposed in Saade et al. (2014)

    Use : y_kmeans,X = saade(A, n_clusters)
        Inputs: A a (symmetric) n x n adjacency sparse matrix of a graph, n_clusters (integer) number of communities
        Outputs: y_kmeans the vector of size n of detected classes, X the eigenvector of size n to use for the classification.

    '''
    
    d = np.sum(A.A, axis = 0)
    n = len(A.A)
    X = np.zeros(n)
    r = np.sqrt(sum(d**2)/sum(d)) # empirical estimate of \sqrt{c\Phi}
    H = (r**2-1)*np.diag(np.ones(n)) + np.diag(d) - r*A
    eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(H, n_clusters, which='SA')
    idx = eigenvalues.argsort()[::-1] # sort the eigenvalues
    idx = idx[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    kmeans = KMeans(n_clusters = n_clusters) # perform kmeans on the informative eigenvector
    X = np.column_stack((X,eigenvectors[:,1:]))
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    
    return y_kmeans,X

