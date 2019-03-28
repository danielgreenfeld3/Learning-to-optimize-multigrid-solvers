import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import numpy as np
import pickle
import time
import tensorflow as tf
import tensorflow.contrib.eager as tfe

def memoize(f):
    memo = {}
    def helper(x):
        if x not in memo:
            memo[x] = f(x)
        return memo[x]
    return helper

def two_d_stencil(num,epsilon=0.0,grid_size=8):
    ##creates the discretization stencil of 2D diffusion
    # problems where the coefficients are drawn from a log-normal distribution.
    epsi = epsilon*np.ones((grid_size,grid_size))
    stencil = np.zeros((num,grid_size,grid_size,3,3))
    diffusion_coeff = np.exp(np.random.normal(size=[num,grid_size, grid_size]))

    #lists of \pm 1 coordinates, modulu gird size
    jm1 = [(i - 1) % grid_size for i in range(grid_size)]
    jp1 = [(i + 1) % grid_size for i in range(grid_size)]

    stencil[:,:, :, 1, 2] = -1. / (6) * (diffusion_coeff[:,jm1] + diffusion_coeff)
    stencil[:,:, :, 2, 1] = -1. / (6) * (diffusion_coeff + diffusion_coeff[:,:, jm1])
    stencil[:,:, :, 2, 0] = -1. / (3) * diffusion_coeff[:,:, jm1]
    stencil[:,:, :, 2, 2] = -1. / (3) * diffusion_coeff
    stencil[:,:, :, 1, 0] = stencil[:,:, jm1, 1, 2]
    stencil[:,:, :, 0, 0] = stencil[:,jm1][:,:,jm1][:,:,:,2, 2]
    stencil[:,:, :, 0, 1] = stencil[:,jm1][:,:,:,2, 1]
    stencil[:,:, :, 0, 2] = stencil[:,jm1][:,:,jp1][:,:,:,2, 0]
    stencil[:,:, :, 1, 1] = -np.sum(np.sum(stencil,axis=4),axis=3) + epsi
    return stencil

def compute_S(A,w=.8,relaxation_type='Jacobi',grid_size=8):
    #computes the iteration matrix of the relaxation, here Gauss-Seidel is used.
    #This function is called on each block seperately.
    n = A.shape[-1]
    B = np.copy(A)
    B[:, np.tril_indices(n, 0)[0],np.tril_indices(n, 0)[1]] = 0. #B is the upper part of A
    batch_size = min(128,A.shape[0])
    res = []
    for i in range(A.shape[0]//batch_size):
        #tf.linalg.triangular_solve ignores the upper part of A, i.e.,
        # A is effectively considered lower triangular
        res.append(tf.linalg.triangular_solve(A[i*batch_size:(i+1)*batch_size],
                                              -B[i*batch_size:(i+1)*batch_size]).numpy())
    return np.stack(res,0)

def map_2_to_1(grid_size=8):
    #maps 2D coordinates to the corresponding 1D coordinate in the matrix.
    k = np.zeros((grid_size,grid_size,3,3))
    M = np.reshape(np.arange(grid_size**2),(grid_size,grid_size)).T
    M = np.concatenate([M,M],0)
    M = np.concatenate([M, M], 1)
    for i in range(3):
        I = (i -1) % grid_size
        for j in range(3):
            J = (j-1) % grid_size
            k[:,:,i,j] = M[I:I+grid_size,J:J+grid_size]
    return k

def compute_A(stencil, tx, ty, ci,grid_size=8):
    #compute the diagonal block of the discretization matrix that correspondes
    #  to the (tx,ty) Fourier mode, using Theorem 1.
    K = map_2_to_1(grid_size=grid_size)
    batch_size = stencil.shape[0]
    A = np.zeros((batch_size,grid_size**2,grid_size**2),dtype=np.complex128)
    X, Y = np.meshgrid(np.arange(-1,2),np.arange(-1,2))
    fourier_component = np.exp(-ci*(tx*X+ty*Y))
    fourier_component = np.reshape(fourier_component,(1,3,3))
    for i in range(grid_size):
        for j in range(grid_size):
            I = int(K[i,j,1,1])
            for k in range(3):
                for m in range(3):
                    J = int(K[i,j,k,m])
                    A[:,I,J] = stencil[:,i,j,k,m] * fourier_component[:,k,m]
    return A

def get_A_S_matrices(num,pi,grid_size,n_size):
    theta_x = np.array(
        [i * 2 * pi / n_size for i in range(-n_size // (2 * grid_size) + 1, n_size // (2 * grid_size) + 1)])
    theta_y = np.array(
        [i * 2 * pi / n_size for i in range(-n_size // (2 * grid_size) + 1, n_size // (2 * grid_size) + 1)])
    A_stencils_test = np.array(two_d_stencil(num))
    A_matrices_test, S_matrices_test = create_matrices(A_stencils_test, grid_size, theta_x, theta_y)
    return A_stencils_test, A_matrices_test, S_matrices_test, len(theta_x)

def create_matrices(A_stencil,grid_size,theta_x,theta_y):
    A_matrices = np.stack(
        [[compute_A(A_stencil, tx, ty, 1j, grid_size=grid_size) for tx in theta_x] for ty in theta_y])
    A_matrices = A_matrices.transpose((2, 0, 1, 3, 4))
    S_matrices = compute_S(A_matrices.reshape((-1, grid_size ** 2, grid_size ** 2)))
    return A_matrices, S_matrices

@memoize
def idx_array(x):
    I, J, batch_size,num_modes = x
    return np.array([[[[i1, i2, ell, I, J] for ell in range(batch_size)] for i2
               in range(num_modes)] for i1
              in range(num_modes)]).reshape(-1, 5).astype(np.int32)

def get_p_matrix_indices(x):
    batch_size, grid_size = x
    K = map_2_to_1(grid_size=grid_size)
    value_indices = []
    indices = []
    for n in range(batch_size):
        for ic in range(grid_size // 2):
            i = 2 * ic
            for jc in range(grid_size // 2):
                j = 2 * jc
                J = int(grid_size // 2 * jc + ic)
                for k in range(3):
                    for m in range(3):
                        I = int(K[i, j, k, m])
                        value_indices.append([n, ic, jc, k, m])
                        indices.append([n,I,J])
    return indices,value_indices
get_p_matrix_indices = memoize(get_p_matrix_indices)

def compute_p2_sparse(P_stencil,n,grid_size):
    with tf.device("gpu:0"):
        indexes,values_indices= get_p_matrix_indices((n,grid_size))
        P = tf.SparseTensor(indexes,tf.gather_nd(P_stencil,values_indices),(n,grid_size**2,(grid_size//2)**2))
        return P

def compute_p2(P_stencil,n,grid_size):
    batch_size = P_stencil.shape[0]
    K = map_2_to_1(grid_size=grid_size)
    pi = np.pi
    theta_x = np.array(([i * 2 * pi / n for i in range(-n // (grid_size*2) + 1, n // (grid_size*2) + 1)]))
    theta_y = np.array([i * 2 * pi / n for i in range(-n // (grid_size*2) + 1, n // (grid_size*2) + 1)])
    num_modes = theta_x.shape[0]

    X, Y = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2))
    with tf.device("gpu:0"):
        P = tf.zeros((len(theta_y), len(theta_x),batch_size,grid_size**2,(grid_size//2)**2),dtype=tf.complex128)
        modes = np.array([[np.exp(-1j * (tx * X + ty * Y)) for tx in theta_x] for ty in theta_y])
        fourier_component = tf.to_complex128(np.tile(modes, (batch_size,1,1,1,1)))
        for ic in range(grid_size//2):
            i = 2*ic #ic is the index on the coarse grid, and i is the index on the fine grid
            for jc in range(grid_size//2):
                j = 2*jc #jc is the index on the coarse grid, and j is the index on the fine grid
                J = int(grid_size//2*jc+ic)
                for k in range(3):
                    for m in range(3):
                        I = int(K[i,j,k,m])
                        a = fourier_component[:,:,:,k, m] * tf.reshape(P_stencil[:, ic, jc, k, m],(-1,1,1))
                        a = tf.transpose(a,(1,2,0))

                        P = P + tf.to_complex128(
                            tf.scatter_nd(tf.constant(idx_array((I,J,int(batch_size),num_modes))),
                                          tf.ones(batch_size*(num_modes**2)),
                                          tf.constant([num_modes,num_modes,batch_size, grid_size**2, (grid_size//2)**2])))*tf.reshape(a, (theta_x.shape[0],theta_y.shape[0], batch_size, 1, 1))
        return P

def create_coarse_training_set(m, pi,num_training_samples, bb=False):
    m.grid_size = 16 #instead of 8
    stencils = []
    additional_num_training_samples = num_training_samples
    theta_x = np.array(
        [i * 2 * pi / 8 for i in range(-8 // (2 * 8) + 1, 8 // (2 * 8) + 1)])
    theta_y = np.array(
        [i * 2 * pi / 8 for i in range(-8 // (2 * 8) + 1, 8 // (2 * 8) + 1)])

    A_stencils_ = two_d_stencil(additional_num_training_samples, grid_size=16)

    batch_size = 128
    for i in range(additional_num_training_samples // batch_size):
        A_matrices_ = np.stack(
            [[compute_A(A_stencils_[i * batch_size:(i + 1) * batch_size], tx, ty, 1j, grid_size=16) for tx in
              theta_x] for ty in theta_y])
        A_matrices_ = A_matrices_.transpose((2, 0, 1, 3, 4))
        A_stencils_temp = tf.convert_to_tensor(A_stencils_[i * batch_size:(i + 1) * batch_size], dtype=tf.double)
        A_matrices__temp = tf.convert_to_tensor(A_matrices_, dtype=tf.complex128)
        A_c, A_c_stencil, _, _ = compute_coarse_matrix(m, 16, A_stencils_temp, A_matrices__temp, 16, bb=bb)
        A_c_stencil = A_c_stencil.numpy()
        stencils.append(A_c_stencil)
    m.grid_size = 8

    return np.concatenate(stencils)


def compute_coarse_matrix(model, n, A_stencil, A_matrices,grid_size,bb=True):
    if bb == True:
        P_stencil = model(A_stencil, True)
    else:
        P_stencil = model(A_stencil, phase="Test")
    P_matrix = tf.to_double(compute_p2_sparse(P_stencil, P_stencil.shape.as_list()[0], grid_size))
    P_matrix_t = tf.sparse_transpose(P_matrix, [0,2, 1])
    A_matrices = tf.squeeze(A_matrices)
    temp = tf.sparse_tensor_to_dense(P_matrix_t)
    q = tf.matmul(temp, tf.to_double(A_matrices))
    A_c = tf.transpose(tf.matmul(temp,tf.transpose(q,[0,2,1])),[0,2,1])
    return A_c, compute_stencil(tf.squeeze(A_c),(grid_size//2)),P_matrix,P_matrix_t

def compute_stencil(A,grid_size):
    indices = get_indices_compute_A((A.shape.as_list()[0],grid_size))
    stencil = tf.reshape(tf.gather_nd(A,indices),(A.shape[0],grid_size,grid_size,3,3))
    return stencil

def get_indices_compute_A(x):
    batch_size,grid_size = x
    indices = []
    K = map_2_to_1(grid_size=grid_size)
    for n in range(batch_size):
        for i in range(grid_size):
            for j in range(grid_size):
                I = int(K[i, j, 1, 1])
                for k in range(3):
                    for m in range(3):
                        J = int(K[i, j, k, m])
                        indices.append([n,I, J])
    return indices
get_indices_compute_A = memoize(get_indices_compute_A)