#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 12:26:17 2023

@author: ayoubbelhadji
"""

import numpy as np
import numpy.polynomial.hermite as herm
from scipy.sparse import diags
from scipy.stats import norm
import math
import ghalton
from scipy import linalg
from tools.SSR_tools import *
from tools.kernel_tools import *


## Quadratures 
def simulate_M(d,M):
    mean = [0]*d
    cov = np.eye(d)
    G = np.random.multivariate_normal(mean, cov, M)
    # G_list = [G[m,:] for m in list(range(M))]
    return G

def Gaussian_kernel(sigma):
    def kernel_aux(x):
        log_output = -((np.linalg.norm(x))**2)/(2*(sigma*sigma))
        return np.exp(log_output)
    return kernel_aux


def sample_unit_sphere(d):
    x = np.random.multivariate_normal(np.zeros(d), np.eye(d))
    y = x/np.linalg.norm(x)
    return y


def MC_on_hypersphere(N,d):
    nodes = [sample_unit_sphere(d) for n in list(range(N))]
    weights = [1/N for n in list(range(N))]
    
    return weights, nodes 


def UG_on_2d_sphere(N,d):
    I = np.linspace(0,1,N+1)
    nodes = [np.asarray([np.cos(2*math.pi*i),np.sin(2*math.pi*i)]) for i in I[:-1]]
    weights = [1/N for n in list(range(N))]
    
    return weights, nodes 

def quasi_monte_carlo_on_2d_sphere(N, d):
    #this method use halton sequence to generate random nodes from Gaussian
    sequencer = ghalton.GeneralizedHalton(d)
    points = np.array(sequencer.get(N))
    M = norm.ppf(points)
    norm_ = np.linalg.norm(M, axis=1)
    uniform = np.diag(1/norm_)@M
    uniform_list = [uniform[i,:] for i in range(N)]
    weights = [1/N for n in list(range(N))]
    return weights, uniform_list

    
def sample_orthogonal_matrix(d):
    # Generate a random matrix with entries from a normal distribution
    random_matrix = np.random.randn(d, d)
    # Perform QR decomposition
    q, r = np.linalg.qr(random_matrix)
    # Ensure the diagonal elements of R are positive
    # This step is to guarantee that the matrix Q is uniformly sampled
    #d = np.diag(r)
    #ph = d / np.abs(d)
    #q *= ph
    return q


def sample_seq_orthogonal_matrices_sym(M,d):
    #input N_S_
    Q =np.zeros((2*M*d,d))
    for m in list(range(M)):
        M_ = sample_orthogonal_matrix(d)
        Q[2*m*d:2*m*d+d,:] = M_
        Q[(2*m+1)*d:(2*m+1)*d+d,:] = -M_
    return Q

def sample_seq_orthogonal_matrices(M,d):
    #input 2*N_S_
    Q =np.zeros((M*d,d))
    for m in list(range(M)):
        M_ = sample_orthogonal_matrix(d)
        Q[m*d:m*d+d,:] = M_
    return Q

def OKQ_on_hypersphere(nodes,d,sigma,reg):
    g_kernel = Gaussian_kernel(sigma)
    N = len(nodes)
    #nodes = [sample_unit_sphere(d) for n in list(range(N))]
    kernel_matrix = np.zeros((N,N))
    
    ones_vector = np.ones((N))
    for i in list(range(N)):
        for j in list(range(N)):
            kernel_matrix[i,j] = g_kernel(nodes[i]-nodes[j])
    #print(len(nodes))
    #print(np.diag(kernel_matrix))
    kernel_matrix = kernel_matrix + reg*np.eye((N))
    # e,v = np.linalg.eigh(kernel_matrix)
    #plt.plot(np.log(e))
    #print(min(e))
    #plt.show()
    weights = linalg.solve(kernel_matrix,ones_vector)
    
    #np.linalg.solve(kernel_matrix, ones_vector)
    #np.dot(np.linalg.inv(kernel_matrix),ones_vector)
    #print(sum(weights))
    return list((1/sum(weights))*weights), nodes



def gaussian_laguerre_quadrature_using_Jacobi(N,alpha):
    ## The method used in this fuction is the one figuring in ???
    diagonals = [[2*n+1+alpha for n in list(range(N))],[np.sqrt((n+1)*(n+1+alpha)) for n in list(range(N-1))],[np.sqrt((n+1)*(n+1+alpha)) for n in list(range(N-1))]]
    J_N = diags(diagonals, [0, -1, 1]).toarray()
    #print(J_N)
    e,v = np.linalg.eigh(J_N)
    #print(v.shape)
    weights = [v[0,i]**2 for i in list(range(N))]
    nodes = [e[i] for i in list(range(N))]
    return weights, nodes



def gaussian_quadrature_using_jacobi_sigma_without_torch(N, sigma_):
    """
    The method used in this function is the one figuring in ???
    :param N: number of nodes
    :param sigma: Gaussian scale
    :return:
    """
    
    sigma = sigma_
    #/np.sqrt(2)
    diagonals = [[0] * N, [np.sqrt(N-n-1) for n in list(range(N-1))], [np.sqrt(N-n-1) for n in list(range(N-1))]]
    J_N = diags(diagonals, [0, -1, 1]).toarray()
    e, v = np.linalg.eigh(J_N)
    weights = [v[N-1, i]**2 for i in list(range(N))]
    nodes = [[sigma*e[i] / 2 for i in list(range(N))]]
#    tensor_nodes = torch.tensor(nodes)
#    tensor_weights = torch.tensor(weights)
#    sigma*np.sqrt(np.pi)*
    return weights, nodes[0]


def combine_radial_spherical_quadratures(r_weights,r_nodes,s_weights,s_nodes):
    weights = []
    nodes = []
    R = len(r_nodes)
    S = len(s_nodes)
    
    for i_r in list(range(R)):
        for i_s in list(range(S)):
            w = r_weights[i_r]*s_weights[i_s]
            n = np.sqrt(2)*(np.sqrt(r_nodes[i_r]))*s_nodes[i_s]
            weights.append(w)
            nodes.append(n)
    

    return np.array(weights), np.array(nodes)
        

## quasi monte carlo

def quasi_monte_carlo_with_halton_nodes(d, N):
    #this method use halton sequence to generate random nodes from Gaussian
    # print('d: ', d, 'N: ', N)
    sequencer = ghalton.GeneralizedHalton(d)
    points = np.array(sequencer.get(int(N)))
    M = norm.ppf(points)
    # M_list = [M[m,:] for m in list(range(len(M)))]
    return M


## sparse hermite

def sparse_gauss_hermite_quadrature(d, n, deg=2):
    #this method subsamples points and weights for Gauss-Hermite quadrature
    #deg: the degree of the polynomial that is approximated accurately by this quadrule is 2*deg-1.
    W, A = herm.hermgauss(deg)

    A = A / A.sum()
    c = np.empty((d, n))
    I = np.arange(W.shape[0])
    for i in range(d):
        c[i] = np.random.choice(I, n, True, A)
    c = c.astype('int')
    W_subsampled = W[c]
    A_subsampled = np.sum(np.log(A[c]), axis=0)
    A_subsampled -= A_subsampled.max()
    A_subsampled = np.exp(A_subsampled)

    A_subsampled /= A_subsampled.sum()
    W_subsampled *= np.sqrt(2)
    # W_list = [W_subsampled[:,m] for m in list(range(len(W_subsampled[1])))]
    return A_subsampled, W_subsampled.T


## MC with orthogonal structure

def MC_with_orthogonal_random_structure(d, N):
    #this method form features W=SQ, 
    #diagonal S follows chi-distribution with degree of freedom d, Q is orthogonal random marix
    D = N
    if D < d:
        G = np.random.randn(d, D)
        Q, _ = np.linalg.qr(G)
        Q = Q.T
    else:
        G = np.random.randn(D, d)
        Q, _ = np.linalg.qr(G)
    d = np.sqrt(2 * np.random.gamma(D/2., 1., D))
    for i in range(Q.shape[0]):
        Q[i, :] *= d[i]
    # Q_list = Q.tolist()
    return Q


## MC with rademacher martrix

def hadamard(d):
    #construct orthogonal Hadamard matrix, each entry has norm n^{-1/2}
    if d < 1:
        lg2 = 0
    else:
        lg2 = int(np.log2(d))
    if 2 ** lg2 != d:
        raise ValueError("d must be an positive integer, and d must be "
                         "a power of 2")

    H = np.zeros((d, d))
    H[0, 0] = 1

    # Sylvester's construction
    for i in range(0, lg2):
        p = 2**i
        H[:p, p:2*p] = H[:p, :p]
        H[p:2*p, :p] = H[:p, :p]
        H[p:2*p, p:2*p] = -H[:p, :p]
    H /= math.sqrt(d)
    return H


def diagonal(d):
    #diagonal matrix D if size n x n with iid Rademacher random variables
    diag = (np.random.randint(0, 2, size=d) - 0.5) * 2
    D = np.diag(diag)
    return D


def single(p, d):
    S = hadamard(d)
    D = diagonal(d)
    M = np.dot(S, D)
    for _ in range(p-1):
        D = diagonal(d)
        M = np.dot(M, np.dot(S, D))
    return M

def MC_with_rademacher_matrix(d, n, p=3):
    '''
    Generates n x n S-Rademacher random matrix as in
        https://arxiv.org/abs/1703.00864.
    '''
    c = np.log2(d)
    f = np.floor(c)
    if f != c:
        d = int(2 ** (f + 1))

    M = np.zeros((n, d))

    t = int(np.ceil(n/d))
    for i in range(t-1):
        M[i*d:(i+1)*d, :] = single(p, d)
    i = t - 1
    M[i*d:, :] = single(p, d)[:n - i*d, :]
    M = np.sqrt(d) * M[:n, :]
    M_list = [M[m,:] for m in list(range(len(M)))]
    return M_list

## 

def kernel_SR_noOKQ_MC(N, d, N_R=3):
    alpha = d/2-1
    N_S_ = N
    N_S = N*d
    r_weights,r_nodes = gaussian_laguerre_quadrature_using_Jacobi(N_R,alpha)
    MCS_weights,MCS_nodes = MC_on_hypersphere(N_S_,d)
    weights, nodes = combine_radial_spherical_quadratures(r_weights,r_nodes,MCS_weights,MCS_nodes)
    return weights, nodes

def kernel_SR_noOKQ_SymOrt(N, d, N_R=3):
    alpha = d/2-1
    N_S_ = int(N/2)
    N_S = 2*N_S_*d
    r_weights,r_nodes = gaussian_laguerre_quadrature_using_Jacobi(N_R,alpha)
    MCS_weights,MCS_nodes = [1/(N_S)]*(N_S), sample_seq_orthogonal_matrices_sym(N_S_,d)

    # MCS_weights,MCS_nodes = MC_on_hypersphere(N_S,d)
    weights, nodes = combine_radial_spherical_quadratures(r_weights,r_nodes,MCS_weights,MCS_nodes)
    return weights, nodes

def kernel_SR_noOKQ_Ort(N, d, N_R=3):
    alpha = d/2-1
    N_S_ = N
    N_S = N*d
    r_weights,r_nodes = gaussian_laguerre_quadrature_using_Jacobi(N_R,alpha)
    MCS_weights,MCS_nodes = [1/(N_S)]*int(N_S), sample_seq_orthogonal_matrices(N_S_,d)

    # MCS_weights,MCS_nodes = MC_on_hypersphere(N_S,d)
    weights, nodes = combine_radial_spherical_quadratures(r_weights,r_nodes,MCS_weights,MCS_nodes)
    return weights, nodes

def kernel_SR_OKQ_MC(N, d, N_R=3, sigma_S = 1):
    ## Quadrature 2: An SR quadrature, the spherical part makes use of i.i.d. samples from the U(Sd), and optimal weights weighted
    ## with respect to the Gaussian kernel (be aware of the spherical bandwidth and the reg parameter)
    alpha = d/2-1
    N_S_ = N
    N_S = N*d
    r_weights,r_nodes = gaussian_laguerre_quadrature_using_Jacobi(N_R,alpha)
    MCS_weights,MCS_nodes = MC_on_hypersphere(N_S_,d)

    sigma_S = 1 #0.2
    lambda_S = 0.00001
        
    OKQMCS_weights, OKQMCS_nodes = OKQ_on_hypersphere(MCS_nodes,d,sigma_S,lambda_S)
    weights, nodes = combine_radial_spherical_quadratures(r_weights,r_nodes,OKQMCS_weights,OKQMCS_nodes)
    return weights, nodes

def kernel_SR_OKQ_SymOrt(N, d, N_R=3, sigma_S = 1):
    ## Quadrature 2: An SR quadrature, the spherical part makes use of i.i.d. samples from the U(Sd), and optimal weights weighted
    ## with respect to the Gaussian kernel (be aware of the spherical bandwidth and the reg parameter)
    alpha = d/2-1
    N_S_ = int(N/2)
    N_S = 2*N_S_*d
    r_weights,r_nodes = gaussian_laguerre_quadrature_using_Jacobi(N_R,alpha)
    MCS_weights,MCS_nodes = [1/(N_S)]*int(N_S), sample_seq_orthogonal_matrices_sym(N_S_,d)

    sigma_S = 1 #0.2
    lambda_S = 0.00001
        
    OKQMCS_weights, OKQMCS_nodes = OKQ_on_hypersphere(MCS_nodes,d,sigma_S,lambda_S)
    weights, nodes = combine_radial_spherical_quadratures(r_weights,r_nodes,OKQMCS_weights,OKQMCS_nodes)
    return weights, nodes

def kernel_SR_OKQ_Ort(N, d, N_R=3, sigma_S = 1):
    ## Quadrature 2: An SR quadrature, the spherical part makes use of i.i.d. samples from the U(Sd), and optimal weights weighted
    ## with respect to the Gaussian kernel (be aware of the spherical bandwidth and the reg parameter)
    alpha = d/2-1
    N_S_ = N
    N_S = N*d
    r_weights,r_nodes = gaussian_laguerre_quadrature_using_Jacobi(N_R,alpha)
    MCS_weights,MCS_nodes = [1/N_S]*(N_S), sample_seq_orthogonal_matrices(N_S_,d)

    sigma_S = 1 #0.2
    lambda_S = 0.00001
        
    OKQMCS_weights, OKQMCS_nodes = OKQ_on_hypersphere(MCS_nodes,d,sigma_S,lambda_S)
    weights_2, nodes_2 = combine_radial_spherical_quadratures(r_weights,r_nodes,OKQMCS_weights,OKQMCS_nodes)
    return weights_2, nodes_2

def kernel_RFF(nn, d, N_R=5):
    N = nn*N_R*d
    weights_MC = [1/N]*N
    nodes_MC = simulate_M(d,N)
    return weights_MC, nodes_MC

def kernel_QMC(nn, d, N_R=5):
    N = nn*N_R*d
    weights_QMC = [1/N]*N
    nodes_QMC = quasi_monte_carlo_with_halton_nodes(d, N)
    return weights_QMC, nodes_QMC

def kernel_GQ(nn, d, N_R=5):
    N = nn*N_R*d
    weights_gq, nodes_gq = sparse_gauss_hermite_quadrature(d, N, deg=2)
    return weights_gq, nodes_gq

def kernel_ORF(nn, d, N_R=5):
    N = nn*N_R*d
    weights_ort = [1/N]*N
    nodes_ort = MC_with_orthogonal_random_structure(d, N)
    return weights_ort, nodes_ort


