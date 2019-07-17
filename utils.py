import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse.linalg
import tensorflow as tf
import scipy
import math
from functools import partial
from tqdm import tqdm
import pyamg
from geometric_solver import geometric_solver


class memoize(object):
    """cache the return value of a method

    This class is meant to be used as a decorator of methods. The return value
    from a given method invocation will be cached on the instance whose method
    was invoked. All arguments passed to a method decorated with memoize must
    be hashable.

    If a memoized method is invoked directly on its class the result will not
    be cached. Instead the method will be invoked like a static method:
    class Obj(object):
        @memoize
        def add_to(self, arg):
            return self + arg
    Obj.add_to(1) # not enough arguments
    Obj.add_to(1, 2) # returns 3, result is not cached
    """

    def __init__(self, func):
        self.func = func

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self.func
        return partial(self, obj)

    def __call__(self, *args, **kw):
        obj = args[0]
        try:
            cache = obj.__cache
        except AttributeError:
            cache = obj.__cache = {}
        key = (self.func, args[1:], frozenset(kw.items()))
        try:
            res = cache[key]
        except KeyError:
            res = cache[key] = self.func(*args, **kw)
        return res


class Utils(object):
    def __init__(self, grid_size=8, device="/cpu:0", bc='dirichlet'):
        self.grid_size = grid_size
        self.device = device
        self.bc = bc

    @staticmethod
    def two_d_stencil_dirichletBC(num, epsilon, grid_size=8):
        epsi = epsilon * np.ones((grid_size, grid_size))
        stencil = np.zeros((num, grid_size, grid_size, 3, 3))

        diffusion_coeff = np.exp(np.random.normal(size=[num, grid_size, grid_size]))

        jm1 = [(i - 1) % grid_size for i in range(grid_size)]
        stencil[:, :, :, 1, 2] = -1. / 6 * (diffusion_coeff[:, jm1] + diffusion_coeff)
        stencil[:, :, :, 2, 1] = -1. / 6 * (diffusion_coeff + diffusion_coeff[:, :, jm1])
        stencil[:, :, :, 2, 0] = -1. / 3 * diffusion_coeff[:, :, jm1]
        stencil[:, :, :, 2, 2] = -1. / 3 * diffusion_coeff

        jp1 = [(i + 1) % grid_size for i in range(grid_size)]

        stencil[:, :, :, 1, 0] = stencil[:, :, jm1, 1, 2]
        stencil[:, :, :, 0, 0] = stencil[:, jm1][:, :, jm1][:, :, :, 2, 2]
        stencil[:, :, :, 0, 1] = stencil[:, jm1][:, :, :, 2, 1]
        stencil[:, :, :, 0, 2] = stencil[:, jm1][:, :, jp1][:, :, :, 2, 0]
        stencil[:, :, :, 1, 1] = -np.sum(np.sum(stencil, axis=4), axis=3) + epsi

        stencil[:, :, 0, :, 0] = 0.
        stencil[:, :, -1, :, -1] = 0.
        stencil[:, 0, :, 0, :] = 0.
        stencil[:, -1, :, -1, :] = 0.
        return stencil

    @staticmethod
    def two_d_stencil_periodicBC(num, epsilon=0.0, epsilon_sparse=False, grid_size=8):
        # creates the discretization stencil of 2D diffusion
        # problems where the coefficients are drawn from a log-normal distribution.

        if not epsilon_sparse:
            epsi = epsilon * np.ones((grid_size, grid_size))
        else:  # a single epsilon value for each grid, to simulate boundaries
            epsilon_coord = np.random.randint(grid_size, size=num)
            epsi = np.zeros((num, grid_size, grid_size))
            for i in range(num):
                epsi[i, epsilon_coord[i], epsilon_coord[i]] = np.exp(np.random.normal(loc=0.0, scale=1.0))
        stencil = np.zeros((num, grid_size, grid_size, 3, 3))
        diffusion_coeff = np.exp(np.random.normal(loc=0.0, scale=1.0, size=[num, grid_size, grid_size]))

        # lists of plus minus  1 coordinates, modulu gird size
        jm1 = [(i - 1) % grid_size for i in range(grid_size)]
        jp1 = [(i + 1) % grid_size for i in range(grid_size)]

        stencil[:, :, :, 1, 2] = -1. / 6 * (diffusion_coeff[:, jm1] + diffusion_coeff)
        stencil[:, :, :, 2, 1] = -1. / 6 * (diffusion_coeff + diffusion_coeff[:, :, jm1])
        stencil[:, :, :, 2, 0] = -1. / 3 * diffusion_coeff[:, :, jm1]
        stencil[:, :, :, 2, 2] = -1. / 3 * diffusion_coeff
        stencil[:, :, :, 1, 0] = stencil[:, :, jm1, 1, 2]
        stencil[:, :, :, 0, 0] = stencil[:, jm1][:, :, jm1][:, :, :, 2, 2]
        stencil[:, :, :, 0, 1] = stencil[:, jm1][:, :, :, 2, 1]
        stencil[:, :, :, 0, 2] = stencil[:, jm1][:, :, jp1][:, :, :, 2, 0]
        stencil[:, :, :, 1, 1] = -np.sum(np.sum(stencil, axis=4), axis=3) + epsi
        return stencil

    def two_d_stencil(self, num, epsilon=0.0, epsilon_sparse=False, grid_size=8):
        if self.bc == 'dirichlet':
            return self.two_d_stencil_dirichletBC(num=num, epsilon=epsilon, grid_size=grid_size)
        elif self.bc == 'periodic':
            return self.two_d_stencil_periodicBC(num=num, epsilon=epsilon, epsilon_sparse=epsilon_sparse, grid_size=grid_size)

    def map_2_to_1(self, grid_size=8):
        # maps 2D coordinates to the corresponding 1D coordinate in the matrix.
        k = np.zeros((grid_size, grid_size, 3, 3))
        M = np.reshape(np.arange(grid_size ** 2), (grid_size, grid_size)).T
        M = np.concatenate([M, M], 0)
        M = np.concatenate([M, M], 1)
        for i in range(3):
            I = (i - 1) % grid_size
            for j in range(3):
                J = (j - 1) % grid_size
                k[:, :, i, j] = M[I:I + grid_size, J:J + grid_size]
        return k

    def compute_S(self, A):
        # computes the iteration matrix of the relaxation, here Gauss-Seidel is used.
        # This function is called on each block seperately.
        n = A.shape[-1]
        B = np.copy(A)
        B[:, np.tril_indices(n, 0)[0], np.tril_indices(n, 0)[1]] = 0.  # B is the upper part of A
        res = []
        for i in range(A.shape[0]):  # range(A.shape[0] // batch_size):
            res.append(scipy.linalg.solve_triangular(a=A[i],
                                                     b=-B[i],
                                                     lower=True, unit_diagonal=False,
                                                     overwrite_b=False, debug=None, check_finite=True))
        return np.stack(res, 0)

    def compute_A(self, stencils, tx, ty, ci, grid_size=8):
        # compute the diagonal block of the discretization matrix that corresponds
        #  to the (tx,ty) Fourier mode, using Theorem 1.
        K = self.map_2_to_1(grid_size=grid_size)
        batch_size = stencils.shape[0]
        A = np.zeros((batch_size, grid_size ** 2, grid_size ** 2), dtype=np.complex128)
        X, Y = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2))
        fourier_component = np.exp(-ci * (tx * X + ty * Y))
        fourier_component = np.reshape(fourier_component, (1, 3, 3))

        for i in range(grid_size):
            for j in range(grid_size):
                I = int(K[i, j, 1, 1])
                for k in range(3):
                    for m in range(3):
                        J = int(K[i, j, k, m])
                        A[:, I, J] = stencils[:, i, j, k, m] * fourier_component[:, k, m]
        return A

    @memoize
    def compute_A_indices(self, grid_size):
        K = self.map_2_to_1(grid_size=grid_size)
        A_idx = []
        stencil_idx = []
        for i in range(grid_size):
            for j in range(grid_size):
                I = int(K[i, j, 1, 1])
                for k in range(3):
                    for m in range(3):
                        J = int(K[i, j, k, m])
                        A_idx.append([I, J])
                        stencil_idx.append([i, j, k, m])
        return np.array(A_idx), stencil_idx

    def compute_csr_matrices(self, stencils, grid_size=8):
        A_idx, stencil_idx = self.compute_A_indices(grid_size)
        if len(stencils.shape) == 5:
            matrices = []
            for stencil in stencils:
                matrices.append(csr_matrix(arg1=(stencil.reshape((-1)), (A_idx[:, 0], A_idx[:, 1])),
                                           shape=(grid_size ** 2, grid_size ** 2)))
            return np.asarray(matrices)
        else:
            return csr_matrix(arg1=(stencils.reshape((-1)), (A_idx[:, 0], A_idx[:, 1])),
                              shape=(grid_size ** 2, grid_size ** 2))

    def compute_p2(self, P_stencil, grid_size):
        indexes = self.get_p_matrix_indices_one(grid_size)
        P = csr_matrix(arg1=(P_stencil.numpy().reshape(-1), (indexes[:, 0], indexes[:, 1])),
                       shape=(grid_size ** 2, (grid_size // 2) ** 2))

        return P

    @memoize
    def get_p_matrix_indices_one(self, grid_size):
        K = self.map_2_to_1(grid_size=grid_size)
        indices = []
        for ic in range(grid_size // 2):
            i = 2 * ic + 1
            for jc in range(grid_size // 2):
                j = 2 * jc + 1
                J = int(grid_size // 2 * jc + ic)
                for k in range(3):
                    for m in range(3):
                        I = int(K[i, j, k, m])
                        indices.append([I, J])

        return np.array(indices)

    def get_A_S_matrices(self, num: int, pi: float, grid_size: int, n_size: int):
        """
        :param num: number of samples to test
        :param pi:
        :param grid_size:
        :param n_size:
        :return:
        """
        theta_x = np.array(
            [i * 2 * pi / n_size for i in range(-n_size // (2 * grid_size) + 1, n_size // (2 * grid_size) + 1)])
        theta_y = np.array(
            [i * 2 * pi / n_size for i in range(-n_size // (2 * grid_size) + 1, n_size // (2 * grid_size) + 1)])
        A_stencils_test = np.array(self.two_d_stencil(num))
        A_matrices_test, S_matrices_test = self.create_matrices(A_stencils_test, grid_size, theta_x, theta_y)
        return A_stencils_test, A_matrices_test, S_matrices_test, len(theta_x)

    def create_matrices(self, A_stencil, grid_size, theta_x, theta_y):
        A_matrices = np.stack(
            [[self.compute_A(A_stencil, tx, ty, 1j, grid_size=grid_size) for tx in theta_x] for ty in theta_y])
        A_matrices = A_matrices.transpose((2, 0, 1, 3, 4))
        S_matrices = self.compute_S(A_matrices.reshape((-1, grid_size ** 2, grid_size ** 2)))
        return A_matrices, S_matrices

    @memoize
    def idx_array(self, x):
        I, J, batch_size, num_modes = x
        return np.array([[[[i1, i2, ell, I, J] for ell in range(batch_size)] for i2
                          in range(num_modes)] for i1
                         in range(num_modes)]).reshape(-1, 5).astype(np.int32)

    @memoize
    def get_p_matrix_indices(self, x):
        batch_size, grid_size = x
        K = self.map_2_to_1(grid_size=grid_size)
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
                            indices.append([n, I, J])
        return indices, value_indices

    def compute_p2_sparse(self, P_stencil, n, grid_size):
        with tf.device(self.device):
            indexes, values_indices = self.get_p_matrix_indices((n, grid_size))
            P = tf.SparseTensor(indices=indexes, values=tf.gather_nd(P_stencil, values_indices),
                                dense_shape=(n, grid_size ** 2, (grid_size // 2) ** 2))
            return P

    def compute_sparse_matrix(self, stencils, batch_size, grid_size):
        with tf.device(self.device):
            indexes, values_indices = self.get_indices_compute_A((batch_size, grid_size))
            tau = tf.SparseTensor(indices=indexes,
                                  values=tf.gather_nd(params=stencils, indices=values_indices),
                                  dense_shape=(batch_size, grid_size ** 2, grid_size ** 2))
            return tau

    def compute_dense_matrix(self, stencils, batch_size, grid_size):
        with tf.device(self.device):
            indexes, values_indices = self.get_indices_compute_A((batch_size, grid_size))
            tau = tf.scatter_nd(indices=indexes,
                                updates=tf.gather_nd(params=stencils, indices=values_indices),
                                shape=(batch_size, grid_size ** 2, grid_size ** 2))
            return tau

    def compute_p2LFA(self, P_stencil, n, grid_size):
        batch_size = P_stencil.get_shape().as_list()[0]
        K = self.map_2_to_1(grid_size=grid_size)
        pi = np.pi
        theta_x = np.array(([i * 2 * pi / n for i in range(-n // (grid_size * 2) + 1, n // (grid_size * 2) + 1)]))
        theta_y = np.array([i * 2 * pi / n for i in range(-n // (grid_size * 2) + 1, n // (grid_size * 2) + 1)])
        num_modes = theta_x.shape[0]

        X, Y = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2))
        with tf.device(self.device):
            P = tf.zeros((len(theta_y), len(theta_x), batch_size, grid_size ** 2, (grid_size // 2) ** 2),
                         dtype=tf.complex128)
            modes = np.array([[np.exp(-1j * (tx * X + ty * Y)) for tx in theta_x] for ty in theta_y])
            fourier_component = tf.to_complex128(np.tile(modes, (batch_size, 1, 1, 1, 1)))
            for ic in range(grid_size // 2):
                i = 2 * ic  # ic is the index on the coarse grid, and i is the index on the fine grid
                for jc in range(grid_size // 2):
                    j = 2 * jc  # jc is the index on the coarse grid, and j is the index on the fine grid
                    J = int(grid_size // 2 * jc + ic)
                    for k in range(3):
                        for m in range(3):
                            I = int(K[i, j, k, m])
                            a = fourier_component[:, :, :, k, m] * tf.reshape(P_stencil[:, ic, jc, k, m], (-1, 1, 1))
                            a = tf.transpose(a, (1, 2, 0))

                            P = P + tf.to_complex128(
                                tf.scatter_nd(indices=tf.constant(self.idx_array((I, J, int(batch_size), num_modes))),
                                              updates=tf.ones(batch_size * (num_modes ** 2)),
                                              shape=tf.constant([num_modes, num_modes, batch_size, grid_size ** 2,
                                                                 (grid_size // 2) ** 2]))) \
                                * tf.reshape(a, (theta_x.shape[0], theta_y.shape[0], batch_size, 1, 1))
            return P

    def compute_stencil(self, A, grid_size):
        if isinstance(A, (tf.Tensor, tf.SparseTensor, tf.Variable)):
            indices, _ = self.get_indices_compute_A((A.shape.as_list()[0], grid_size))
            stencil = tf.reshape(tf.gather_nd(A, indices), (A.shape[0], grid_size, grid_size, 3, 3))
            return stencil
        else:
            indices = self.get_indices_compute_A_one(grid_size)
            stencil = np.array(A[indices[:, 0], indices[:, 1]]).reshape((grid_size, grid_size, 3, 3))
            return tf.to_double(stencil)

    @memoize
    def get_indices_compute_A_one(self, grid_size):
        indices = []
        K = self.map_2_to_1(grid_size=grid_size)
        for i in range(grid_size):
            for j in range(grid_size):
                I = int(K[i, j, 1, 1])
                for k in range(3):
                    for m in range(3):
                        J = int(K[i, j, k, m])
                        indices.append([I, J])

        return np.array(indices)

    @memoize
    def get_indices_compute_A(self, x):
        batch_size, grid_size = x
        indices = []
        value_indices = []
        K = self.map_2_to_1(grid_size=grid_size)
        for n in range(batch_size):
            for i in range(grid_size):
                for j in range(grid_size):
                    I = int(K[i, j, 1, 1])
                    for k in range(3):
                        for m in range(3):
                            J = int(K[i, j, k, m])
                            indices.append([n, I, J])
                            value_indices.append([n, i, j, k, m])
        return indices, value_indices

    def compute_coarse_matrix(self, model, A_stencil, A_matrices, grid_size, bb=True):
        with tf.device(self.device):
            if bb == True:
                P_stencil = model(inputs=A_stencil, black_box=True)
            else:
                P_stencil = model(inputs=A_stencil, black_box=False, phase="Test")
        P_matrix = self.compute_p2(P_stencil, grid_size)
        P_matrix_t = P_matrix.transpose()
        A_c = P_matrix_t @ A_matrices @ P_matrix
        return A_c, self.compute_stencil(A_c, (grid_size // 2)), P_matrix, P_matrix_t

    def compute_coarse_matrixLFA(self, model, n, A_stencil, A_matrices, grid_size, bb=True):
        with tf.device(self.device):
            if bb == True:
                P_stencil = model(inputs=A_stencil, black_box=True)
            else:
                P_stencil = model(inputs=A_stencil, black_box=False, phase="Test")
        P_matrix = (self.compute_p2LFA(P_stencil, n, grid_size)).numpy()
        P_matrix_t = (tf.transpose(P_matrix)).numpy()
        A_c = P_matrix_t @ A_matrices @ P_matrix
        return A_c, self.compute_stencil(A_c, (grid_size // 2)), P_matrix, P_matrix_t

    def compute_coarse_matrix_sparse(self, model, A_stencil, A_matrices, grid_size, bb=True):
        if bb == True:
            P_stencil = model(inputs=A_stencil, black_box=True)
        else:
            P_stencil = model(inputs=A_stencil, black_box=False, phase="Test")
        P_matrix = tf.to_double(self.compute_p2_sparse(P_stencil, P_stencil.shape.as_list()[0], grid_size))
        P_matrix_t = tf.sparse_transpose(P_matrix, [0, 2, 1])
        A_matrices = tf.squeeze(A_matrices)
        temp = tf.sparse_tensor_to_dense(P_matrix_t)
        q = tf.matmul(temp, tf.to_double(A_matrices))
        A_c = tf.transpose(tf.matmul(temp, tf.transpose(q, [0, 2, 1])), [0, 2, 1])
        return A_c, self.compute_stencil(tf.squeeze(A_c), (grid_size // 2)), P_matrix, P_matrix_t

    def create_coarse_training_set(self, m, pi, num_training_samples, bb=False, epsilon_sparse=False):
        m.grid_size = 16  # instead of 8
        stencils = []
        additional_num_training_samples = num_training_samples
        theta_x = np.array(
            [i * 2 * pi / 8 for i in range(-8 // (2 * 8) + 1, 8 // (2 * 8) + 1)])
        theta_y = np.array(
            [i * 2 * pi / 8 for i in range(-8 // (2 * 8) + 1, 8 // (2 * 8) + 1)])

        A_stencils_ = self.two_d_stencil(additional_num_training_samples, grid_size=16, epsilon_sparse=epsilon_sparse)

        batch_size = 128
        for i in tqdm(range(additional_num_training_samples // batch_size)):
            A_matrices_ = np.stack(
                [[self.compute_A(A_stencils_[i * batch_size:(i + 1) * batch_size], tx, ty, 1j, grid_size=16) for tx in
                  theta_x] for ty in theta_y])
            A_matrices_ = A_matrices_.transpose((2, 0, 1, 3, 4))
            A_stencils_temp = tf.convert_to_tensor(A_stencils_[i * batch_size:(i + 1) * batch_size], dtype=tf.double)
            A_matrices__temp = tf.convert_to_tensor(A_matrices_, dtype=tf.complex128)
            A_c, A_c_stencil, _, _ = self.compute_coarse_matrix_sparse(m, A_stencils_temp, A_matrices__temp, 16,
                                                                       bb=bb)
            A_c_stencil = A_c_stencil.numpy()
            stencils.append(A_c_stencil)
        m.grid_size = 8

        return np.concatenate(stencils)

    def mg_levels(self, model, n, A_stencil, A_matrices, grid_size, max_depth=3, bb=False):
        res = {'A0': A_matrices}
        for i in range(max_depth):
            A_matrices, A_stencil, P, _ = self.compute_coarse_matrix(model, n // (2 ** i), A_stencil,
                                                                     A_matrices, grid_size // (2 ** i), bb=bb)
            model.grid_size = model.grid_size // 2
            A_stencil = tf.convert_to_tensor([A_stencil])
            res['A' + str(i + 1)] = A_matrices
            res['P' + str(i)] = P
        return res

    def solve_with_model(self, model, A_matrices, b, initial_guess, max_iterations, max_depth=3, blackbox=False,
                         w_cycle=False):
        def prolongation_fn(A, args):
            is_blackbox = args["is_blackbox"]
            grid_size = int(math.sqrt(A.shape[0]))
            indices = self.get_indices_compute_A_one(grid_size)
            A_stencil = np.array(A[indices[:, 0], indices[:, 1]]).reshape((grid_size, grid_size, 3, 3))
            model.grid_size = grid_size  # TODO: infer grid_size automatically

            tf_A_stencil = tf.convert_to_tensor([A_stencil])
            with tf.device(self.device):
                if is_blackbox:
                    P_stencil = model(inputs=tf_A_stencil, black_box=True)
                else:
                    P_stencil = model(inputs=tf_A_stencil, black_box=False, phase="Test")
            return self.compute_p2(P_stencil, grid_size).astype(np.double)  # imaginary part should be zero

        prolongation_args = {"is_blackbox": blackbox}

        error_norms = []

        #  solver calls this function after each iteration
        def error_callback(x_k):
            error_norms.append(pyamg.util.linalg.norm(x_k))

        solver = geometric_solver(A_matrices, prolongation_fn, prolongation_args,
                                  max_levels=max_depth)

        if w_cycle:
            cycle = 'W'
        else:
            cycle = 'V'
        residual_norms = []
        x = solver.solve(b, x0=initial_guess, maxiter=max_iterations, cycle=cycle, residuals=residual_norms, tol=0,
                         callback=error_callback)
        return x, residual_norms, error_norms, solver
