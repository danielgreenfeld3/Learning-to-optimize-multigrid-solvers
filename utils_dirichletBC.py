from scipy.sparse import csr_matrix,lil_matrix
import scipy.sparse.linalg
from scipy.io import  savemat
import math
from multiprocessing import Pool
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import numpy as np
import pickle
import time
import tensorflow as tf

import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()

def memoize(f):
    memo = {}
    def helper(x):
        if x not in memo:
            memo[x] = f(x)
        return memo[x]
    return helper

def two_d_stencil(grid_size, epsilon):
    epsi = epsilon * np.ones((grid_size, grid_size))
    stencil = np.zeros((grid_size,grid_size,3,3))

    diffusion_coeff = np.exp(np.random.normal(size=[grid_size, grid_size]))
    # diffusion_coeff = np.random.uniform(size=[grid_size, grid_size])
    # diffusion_coeff = np.ones(shape=[grid_size, grid_size])

    jm1 = [(i - 1) % grid_size for i in range(grid_size)]
    stencil[:, :, 1, 2] = -1. / (6) * (diffusion_coeff[jm1] + diffusion_coeff)
    stencil[:, :, 2, 1] = -1. / (6) * (diffusion_coeff + diffusion_coeff[:, jm1])
    stencil[:, :, 2, 0] = -1. / (3) * diffusion_coeff[:, jm1]
    stencil[:, :, 2, 2] = -1. / (3) * diffusion_coeff

    jm1 = [(i-1)%grid_size for i in range(grid_size)]
    jp1 = [(i + 1) % grid_size for i in range(grid_size)]
    stencil[:, :, 1, 0] = stencil[:, jm1, 1, 2]
    stencil[:, :, 0, 0] = stencil[jm1][:,jm1][:,:,2, 2]
    stencil[:, :, 0, 1] = stencil[jm1,:, 2, 1]
    stencil[:, :, 0, 2] = stencil[jm1][:,jp1][:,:,2, 0]
    stencil[:, :, 1, 1] = -np.sum(np.sum(stencil,axis=3),axis=2) + epsi

    stencil[:, 0, :, 0] = 0.
    stencil[:, -1, :, -1] = 0.
    stencil[0, :, 0, :] = 0.
    stencil[-1, :, -1, :] = 0.
    return stencil

def map_2_to_1(grid_size=8):
    k = np.zeros((grid_size,grid_size,3,3))
    M = np.reshape(np.arange(grid_size**2),(grid_size,grid_size)).T
    M = np.concatenate([M,M],0)
    M = np.concatenate([M, M], 1)
    for i in range(3):
        I = (i -1 )% grid_size
        for j in range(3):
            J = (j-1)%grid_size
            k[:,:,i,j] = M[I:I+grid_size,J:J+grid_size]
    return k

def compute_A_indecis(grid_size):
    K = map_2_to_1(grid_size=grid_size)
    A_idx = []
    stencil_idx = []
    for i in range(grid_size):
        for j in range(grid_size):
            I = int(K[i,j,1,1])
            for k in range(3):
                for m in range(3):
                    J = int(K[i,j,k,m])
                    A_idx.append([I,J])
                    stencil_idx.append([i,j,k,m])
    return np.array(A_idx), stencil_idx

compute_A_indecis = memoize(compute_A_indecis)

def compute_A(stencil, grid_size=8):
    A_idx, stencil_idx = compute_A_indecis(grid_size)
    A = csr_matrix((stencil.reshape((-1)),(A_idx[:,0],A_idx[:,1])),shape=(grid_size**2,grid_size**2))
    return A

def compute_stencil(A,grid_size):
    indices = get_indices_compute_A(grid_size)
    stencil = np.array(A[indices[:,0],indices[:,1]]).reshape(grid_size,grid_size,3,3)
    return tf.to_double(stencil)

def idx_array(x):
    I, J, batch_size,num_modes = x
    return np.array([[[[i1, i2, ell, I, J] for ell in range(batch_size)] for i2
               in range(num_modes)] for i1
              in range(num_modes)]).reshape(-1, 5).astype(np.int32)
idx_array = memoize(idx_array)

def get_indices_compute_A(grid_size):
    indices = []
    K = map_2_to_1(grid_size=grid_size)
    for i in range(grid_size):
        for j in range(grid_size):
            I = int(K[i, j, 1, 1])
            for k in range(3):
                for m in range(3):
                    J = int(K[i, j, k, m])
                    indices.append([I, J])
    return np.array(indices)
get_indices_compute_A = memoize(get_indices_compute_A)

def get_p_matrix_indices(grid_size):
    K = map_2_to_1(grid_size=grid_size)
    value_indices = []
    indices = []
    for ic in range(grid_size // 2):
        i = 2 * ic + 1
        for jc in range(grid_size // 2):
            j = 2 * jc + 1
            J = int(grid_size // 2 * jc + ic)
            for k in range(3):
                for m in range(3):
                    I = int(K[i, j, k, m])
                    value_indices.append([0, ic, jc, k, m])
                    indices.append([I,J])
    return np.array(indices),np.array(value_indices)
get_p_matrix_indices = memoize(get_p_matrix_indices)

def compute_p2(P_stencil,n,grid_size):
    indexes,values_indices = get_p_matrix_indices(grid_size)
    P = csr_matrix((P_stencil.numpy().reshape(-1), (indexes[:,0],indexes[:,1])),
                   shape=(grid_size**2,(grid_size//2)**2))
    return P

def compute_coarse_matrix(model, n, A_stencil, A_matrices,grid_size,bb=True):
    with tf.device('gpu:0'):
        if bb == True:
            P_stencil = model(A_stencil, True)
        else:
            P_stencil = model(A_stencil, phase="Test")
    P_matrix = compute_p2(P_stencil, n, grid_size)
    P_matrix_t = P_matrix.transpose()
    A_c = P_matrix_t@A_matrices@P_matrix
    return A_c, compute_stencil(A_c,(grid_size//2)),P_matrix,P_matrix_t

def mg_levels(model, n, A_stencil, A_matrices, grid_size, max_depth=3, bb=False):
    res = {'A0':A_matrices}
    for i in range(max_depth):
        A_matrices, A_stencil, P,_ = compute_coarse_matrix(model, n//(2**i), A_stencil,
                                                            A_matrices, grid_size//(2**i), bb=bb)
        model.grid_size = model.grid_size//2
        A_stencil = tf.convert_to_tensor([A_stencil])
        res['A'+str(i+1)] = A_matrices
        res['P' + str(i)] = P
    return res

def relax(A,x,b):
    L = scipy.sparse.tril(A,format='csr')
    U = scipy.sparse.triu(A,1,format='csr')
    return scipy.sparse.linalg.spsolve_triangular(L,b-U@x)

def cycle(model, n, A_stencil, A_matrices, b, initial_guess, grid_size, depth, max_depth=3, bb=False, cache=[], w_cycle=False):
    update_cache = len(cache) == 0
    if depth==max_depth:
        sol = scipy.sparse.linalg.spsolve(A_matrices,b)
        return sol,cache
    else:
        if len(cache) == 0:
            A_c, A_c_stencil, P, R = compute_coarse_matrix(model, n, A_stencil, A_matrices, grid_size, bb=bb)
            A_c_stencil = tf.convert_to_tensor([A_c_stencil])
        else:
            A_c, A_c_stencil, P, R = cache[-1]
        if initial_guess is not None:
            x_current = initial_guess
        else:
            x_current = np.zeros((grid_size ** 2,1))
        x_current = relax(A_matrices, x_current, b)
        res = b - A_matrices@x_current
        restricted_res = R@res
        model.grid_size = model.grid_size//2
        coarse_sol,cache_ = cycle(model, n // 2, A_c_stencil, A_c, restricted_res, None, grid_size // 2,
                                  depth + 1, max_depth=max_depth, bb=bb, cache=cache[:-1],w_cycle=w_cycle)
        if w_cycle:
            coarse_sol,_ = cycle(model, n // 2, A_c_stencil, A_c, restricted_res, coarse_sol, grid_size // 2,
                                 depth + 1, max_depth=max_depth, bb=bb,cache=cache[:-1],w_cycle=w_cycle)
        model.grid_size = model.grid_size*2+1
        x_current = x_current + P@coarse_sol
        x_current = relax(A_matrices, x_current, b)

        if depth == 0:
            print(np.linalg.norm(x_current))
        if update_cache:
            cache_.append([A_c, A_c_stencil, P, R ])
            cache = cache_
        return x_current,cache