import os
import numpy as np
import tensorflow as tf
import argparse
import random
import string
from utils import Utils
from tqdm import tqdm
from tensorboardX import SummaryWriter

tf.enable_eager_execution()

DEVICE = "/cpu:0"

num_training_samples = 10 * 16384
num_test_samples = 128
grid_size = 8
n_test, n_train = 32, 8
checkpoint_dir = './training_dir'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')

with tf.device(DEVICE):
    lr_ = 1.2e-5
    lr = tf.Variable(lr_)
    optimizer = tf.train.AdamOptimizer(lr)


def loss(model, n, A_stencil, A_matrices, S_matrices, index=None, pos=-1., phase="Training", epoch=-1, grid_size=8,
         remove=True):
    with tf.device(DEVICE):
        A_matrices = tf.conj(A_matrices)
        S_matrices = tf.conj(S_matrices)
        pi = tf.constant(np.pi)
        theta_x = np.array(([i * 2 * pi / n for i in range(-n // (grid_size * 2) + 1, n // (grid_size * 2) + 1)]))
    with tf.device(DEVICE):
        if phase == "Test" and epoch == 0:
            P_stencil = model(A_stencil, True)
            P_matrix = utils.compute_p2LFA(P_stencil, n, grid_size)
            P_matrix = tf.transpose(P_matrix, [2, 0, 1, 3, 4])
            P_matrix_t = tf.transpose(P_matrix, [0, 1, 2, 4, 3], conjugate=True)
            A_c = tf.matmul(tf.matmul(P_matrix_t, A_matrices), P_matrix)

            index_to_remove = len(theta_x) * (-1 + n // (2 * grid_size)) + n // (2 * grid_size) - 1
            A_c = tf.reshape(A_c, (-1, int(theta_x.shape[0]) ** 2, (grid_size // 2) ** 2, (grid_size // 2) ** 2))
            A_c_removed = tf.concat([A_c[:, :index_to_remove], A_c[:, index_to_remove + 1:]], 1)
            P_matrix_t_reshape = tf.reshape(P_matrix_t,
                                            (-1, int(theta_x.shape[0]) ** 2, (grid_size // 2) ** 2, grid_size ** 2))
            P_matrix_reshape = tf.reshape(P_matrix,
                                          (-1, int(theta_x.shape[0]) ** 2, grid_size ** 2, (grid_size // 2) ** 2))
            A_matrices_reshaped = tf.reshape(A_matrices,
                                             (-1, int(theta_x.shape[0]) ** 2, grid_size ** 2, grid_size ** 2))
            A_matrices_removed = tf.concat(
                [A_matrices_reshaped[:, :index_to_remove], A_matrices_reshaped[:, index_to_remove + 1:]], 1)

            P_matrix_removed = tf.concat(
                [P_matrix_reshape[:, :index_to_remove], P_matrix_reshape[:, index_to_remove + 1:]], 1)
            P_matrix_t_removed = tf.concat(
                [P_matrix_t_reshape[:, :index_to_remove], P_matrix_t_reshape[:, index_to_remove + 1:]], 1)

            A_coarse_inv_removed = tf.matrix_solve(A_c_removed, P_matrix_t_removed)

            CGC_removed = tf.eye(grid_size ** 2, dtype=tf.complex128) \
                          - tf.matmul(tf.matmul(P_matrix_removed, A_coarse_inv_removed), A_matrices_removed)
            S_matrices_reshaped = tf.reshape(S_matrices,
                                             (-1, int(theta_x.shape[0]) ** 2, grid_size ** 2, grid_size ** 2))
            S_removed = tf.concat(
                [S_matrices_reshaped[:, :index_to_remove], S_matrices_reshaped[:, index_to_remove + 1:]], 1)
            iteration_matrix = tf.matmul(tf.matmul(CGC_removed, S_removed), S_removed)
            loss_test = tf.reduce_mean(tf.reduce_mean(tf.reduce_sum(tf.square(tf.abs(iteration_matrix)), [2, 3]), 1))
            return tf.constant([0.]), loss_test.numpy()
        if index is not None:
            P_stencil = model(A_stencil, index=index, pos=pos, phase=phase)
        else:
            P_stencil = model(A_stencil, phase=phase)

        if not (phase == "Test" and epoch == 0):
            P_matrix = utils.compute_p2LFA(P_stencil, n, grid_size)

            P_matrix = tf.transpose(P_matrix, [2, 0, 1, 3, 4])
            P_matrix_t = tf.transpose(P_matrix, [0, 1, 2, 4, 3], conjugate=True)

            A_c = tf.matmul(tf.matmul(P_matrix_t, A_matrices), P_matrix)
            index_to_remove = len(theta_x) * (-1 + n // (2 * grid_size)) + n // (2 * grid_size) - 1
            A_c = tf.reshape(A_c, (-1, int(theta_x.shape[0]) ** 2, (grid_size // 2) ** 2, (grid_size // 2) ** 2))
            A_c_removed = tf.concat([A_c[:, :index_to_remove], A_c[:, index_to_remove + 1:]], 1)
            P_matrix_t_reshape = tf.reshape(P_matrix_t,
                                            (-1, int(theta_x.shape[0]) ** 2, (grid_size // 2) ** 2, grid_size ** 2))
            P_matrix_reshape = tf.reshape(P_matrix,
                                          (-1, int(theta_x.shape[0]) ** 2, grid_size ** 2, (grid_size // 2) ** 2))
            A_matrices_reshaped = tf.reshape(A_matrices,
                                             (-1, int(theta_x.shape[0]) ** 2, grid_size ** 2, grid_size ** 2))
            A_matrices_removed = tf.concat(
                [A_matrices_reshaped[:, :index_to_remove], A_matrices_reshaped[:, index_to_remove + 1:]], 1)

            P_matrix_removed = tf.concat(
                [P_matrix_reshape[:, :index_to_remove], P_matrix_reshape[:, index_to_remove + 1:]], 1)
            P_matrix_t_removed = tf.concat(
                [P_matrix_t_reshape[:, :index_to_remove], P_matrix_t_reshape[:, index_to_remove + 1:]], 1)
            A_coarse_inv_removed = tf.matrix_solve(A_c_removed, P_matrix_t_removed)

            CGC_removed = tf.eye(grid_size ** 2, dtype=tf.complex128) \
                          - tf.matmul(tf.matmul(P_matrix_removed, A_coarse_inv_removed), A_matrices_removed)
            S_matrices_reshaped = tf.reshape(S_matrices,
                                             (-1, int(theta_x.shape[0]) ** 2, grid_size ** 2, grid_size ** 2))
            S_removed = tf.concat(
                [S_matrices_reshaped[:, :index_to_remove], S_matrices_reshaped[:, index_to_remove + 1:]], 1)
            iteration_matrix_all = tf.matmul(tf.matmul(CGC_removed, S_removed), S_removed)

            if remove:
                if phase != 'Test':
                    iteration_matrix = iteration_matrix_all
                    for _ in range(0):
                        iteration_matrix = tf.matmul(iteration_matrix_all, iteration_matrix_all)
                else:
                    iteration_matrix = iteration_matrix_all
                loss = tf.reduce_mean(
                    tf.reduce_max(tf.pow(tf.reduce_sum(tf.square(tf.abs(iteration_matrix)), [2, 3]), 1), 1))
            else:
                loss = tf.reduce_mean(
                    tf.reduce_mean(tf.reduce_sum(tf.square(tf.abs(iteration_matrix_all)), [2, 3]), 1))

                print("Real loss: ", loss.numpy())
            real_loss = loss.numpy()
            return loss, real_loss


def grad(model, n, A_stencil, A_matrices, S_matrices, phase="Training", epoch=-1, grid_size=8, remove=True):
    with tf.GradientTape() as tape:
        loss_value, real_loss = loss(model, n, A_stencil, A_matrices, S_matrices,
                                     phase=phase, epoch=epoch, grid_size=grid_size, remove=remove)
    return tape.gradient(loss_value, m.variables), real_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true', help="")
    parser.add_argument('--use-gpu', action='store_true', default=True, help="")
    parser.add_argument('--grid-size', default=8, type=int, help="")
    parser.add_argument('--batch-size', default=32, type=int, help="")
    parser.add_argument('--n-epochs', default=2, type=int, help="")
    parser.add_argument('--bc', default='periodic')

    args = parser.parse_args()

    if args.use_gpu:
        DEVICE = "/gpu:0"

    utils = Utils(grid_size=args.grid_size, device=DEVICE, bc=args.bc)

    random_string = ''.join(random.choices(string.digits, k=5))  # to make the run_name string unique
    run_name = f"regularization_grid_size={args.grid_size}_batch_size={args.batch_size}_{random_string}"
    writer = SummaryWriter(log_dir='runs/' + run_name)

    if args.bc == 'periodic':
        from model_periodicBC import Pnetwork
    else:
        from model_dirichletBC import Pnetwork

    # define network
    m = Pnetwork(grid_size=grid_size, device=DEVICE)

    root = tf.train.Checkpoint(optimizer=optimizer, model=m, optimizer_step=tf.train.get_or_create_global_step())

    with tf.device(DEVICE):
        pi = tf.constant(np.pi)
        ci = tf.to_complex128(1j)

    A_stencils_test, A_matrices_test, S_matrices_test, num_of_modes = utils.get_A_S_matrices(num_test_samples, np.pi,
                                                                                             grid_size, n_test)

    with tf.device(DEVICE):
        A_stencils_test = tf.convert_to_tensor(A_stencils_test, dtype=tf.double)
        A_matrices_test = tf.convert_to_tensor(A_matrices_test, dtype=tf.complex128)
        S_matrices_test = tf.reshape(tf.convert_to_tensor(S_matrices_test, dtype=tf.complex128),
                                     (-1, num_of_modes, num_of_modes, grid_size ** 2, grid_size ** 2))

    A_stencils_train = np.array(utils.two_d_stencil(num_training_samples))
    n_train_list = [16, 16, 32]
    initial_epsi = 1e-0

    numiter = -1
    for j in range(len(n_train_list)):
        A_stencils = A_stencils_train.copy()
        n_train = n_train_list[j]

        theta_x = np.array(
            [i * 2 * pi / n_train for i in range(-n_train // (2 * grid_size) + 1, n_train // (2 * grid_size) + 1)])
        theta_y = np.array(
            [i * 2 * pi / n_train for i in range(-n_train // (2 * grid_size) + 1, n_train // (2 * grid_size) + 1)])

        for epoch in range(args.n_epochs):
            print("epoch: {}".format(epoch))
            order = np.random.permutation(num_training_samples)

            _, blackbox_test_loss = grad(model=m, n=n_test, A_stencil=A_stencils_test,
                                         A_matrices=A_matrices_test, S_matrices=S_matrices_test,
                                         phase="Test", epoch=0, grid_size=grid_size)

            if epoch % 1 == 0:  # change to save once every X epochs
                root.save(file_prefix=checkpoint_prefix)

            for iter in tqdm(range(num_training_samples // args.batch_size)):
                numiter += 1
                idx = np.random.choice(A_stencils.shape[0], args.batch_size, replace=False)
                A_matrices = np.stack(
                    [[utils.compute_A(A_stencils[idx], tx, ty, 1j, grid_size=grid_size) for tx in theta_x] for ty in
                     theta_y])
                A_matrices = A_matrices.transpose((2, 0, 1, 3, 4))

                S_matrices = np.reshape(utils.compute_S(A_matrices.reshape((-1, grid_size ** 2, grid_size ** 2))),
                                        (-1, theta_x.shape[0], theta_x.shape[0], grid_size ** 2, grid_size ** 2))
                with tf.device(DEVICE):
                    A_stencils_tensor = tf.convert_to_tensor(A_stencils[idx], dtype=tf.double)
                    A_matrices_tensor = tf.convert_to_tensor(A_matrices, dtype=tf.complex128)
                    S_matrices_tensor = tf.convert_to_tensor(S_matrices, dtype=tf.complex128)

                _, blackbox_train_loss = grad(m, n_train, A_stencils_tensor, A_matrices_tensor, S_matrices_tensor,
                                              epoch=0,
                                              grid_size=grid_size, remove=True, phase="Test")
                grads, real_loss = grad(m, n_train, A_stencils_tensor, A_matrices_tensor,
                                        S_matrices_tensor, grid_size=grid_size, remove=True, phase="p")
                writer.add_scalar('loss', real_loss, numiter)
                writer.add_scalar('blackbox_train_loss', blackbox_train_loss, numiter)
                writer.add_scalar('blackbox_test_loss', blackbox_test_loss, numiter)
                optimizer.apply_gradients(zip(grads, m.variables), tf.train.get_or_create_global_step())

        # add coarse grid problems:
        if j > 0:
            num_training_samples = num_training_samples // 2
        temp = utils.create_coarse_training_set(m, pi, num_training_samples)
        A_stencils_train = np.concatenate(
            [np.array(utils.two_d_stencil(num_training_samples)), temp], axis=0)
        num_training_samples = A_stencils_train.shape[0]
