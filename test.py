import tensorflow as tf
import numpy as np
from utils import Utils
import matplotlib.pyplot as plt
import argparse
import scipy
from tqdm import tqdm

tf.enable_eager_execution()
DEVICE = '/gpu:0'
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # the grid should be 2^n-1
    parser.add_argument('--grid-size', default=255, type=int, help="")
    parser.add_argument('--num-test-samples', default=100, type=int, help="")
    parser.add_argument('--boundary', default='dirichlet', type=str, help="")
    parser.add_argument('--compute-spectral-radius', default=False, type=bool, help="")
    parser.add_argument('--bb-row-normalize', default=False, type=bool, help="")

    args = parser.parse_args()

    num_cycles = 41
    utils = Utils(grid_size=args.grid_size, device=DEVICE, bc=args.boundary)
    if args.boundary == 'dirichlet':
        from model_dirichletBC import Pnetwork
    else:
        from model_periodicBC import Pnetwork
    m = Pnetwork(grid_size=args.grid_size, device=DEVICE)

    checkpoint_dir = './training_dir'

    with tf.device(DEVICE):
        lr = tf.Variable([3.4965356e-05])
        optimizer = tf.train.AdamOptimizer(lr)
    root = tf.train.Checkpoint(optimizer=optimizer, model=m, optimizer_step=tf.train.get_or_create_global_step())
    root.restore(tf.train.latest_checkpoint(checkpoint_dir))

    black_box_residual_norms = []
    black_box_errors = []
    black_box_frob_norms = []
    black_box_spectral_radii = []
    net_residual_norms = []
    net_errors = []
    network_spectral_radii = []
    network_frob_norms = []
    A_stencils_test = utils.two_d_stencil(num=args.num_test_samples, grid_size=args.grid_size, epsilon=0.0)
    for A_stencil in tqdm(A_stencils_test):
        A_matrix = utils.compute_csr_matrices(stencils=A_stencil, grid_size=args.grid_size)

        A_stencil_tf = tf.convert_to_tensor(value=[A_stencil], dtype=tf.double)
        b = np.zeros(shape=(args.grid_size ** 2, 1))

        initial = np.random.normal(loc=0.0, scale=1.0, size=args.grid_size ** 2)
        initial = initial[:, np.newaxis]

        _, residual_norms, error_norms, solver = utils.solve_with_model(model=m,
                                                                        A_matrices=A_matrix, b=b,
                                                                        initial_guess=initial,
                                                                        max_iterations=num_cycles,
                                                                        max_depth=int(np.log2(args.grid_size)) - 1,
                                                                        blackbox=True,
                                                                        w_cycle=True)
        black_box_errors.append(error_norms)
        black_box_residual_norms.append(residual_norms)
        if args.compute_spectral_radius:
            I = np.eye(args.grid_size ** 2, dtype=np.double)
            P = solver.levels[0].P
            R = solver.levels[0].R
            A = solver.levels[0].A
            C = I - P @ scipy.sparse.linalg.inv(R @ A @ P) @ R @ A

            L = scipy.sparse.tril(A)
            S = I - scipy.sparse.linalg.inv(L) @ A
            M = S @ C @ S
            black_box_frob_norms.append(scipy.linalg.norm(M))
            eigs, _ = scipy.sparse.linalg.eigs(M)
            black_box_spectral_radius = eigs.max()
            black_box_spectral_radii.append(black_box_spectral_radius)

        x, residual_norms, error_norms, solver = utils.solve_with_model(model=m,
                                                                        A_matrices=A_matrix, b=b, initial_guess=initial,
                                                                        max_iterations=num_cycles,
                                                                        max_depth=int(np.log2(args.grid_size)) - 1,
                                                                        blackbox=False,
                                                                        w_cycle=True)

        net_errors.append(error_norms)
        net_residual_norms.append(residual_norms)
        if args.compute_spectral_radius:
            I = np.eye(args.grid_size ** 2, dtype=np.double)
            P = solver.levels[0].P
            R = solver.levels[0].R
            A = solver.levels[0].A
            C = I - P @ scipy.sparse.linalg.inv(R @ A @ P) @ R @ A

            L = scipy.sparse.tril(A)
            S = I - scipy.sparse.linalg.inv(L) @ A
            M = S @ C @ S
            network_frob_norms.append(scipy.linalg.norm(M))
            eigs, _ = scipy.sparse.linalg.eigs(M)
            network_spectral_radius = eigs.max()
            network_spectral_radii.append(network_spectral_radius)

    net_errors = np.array(net_errors)
    net_errors_log = np.log2(net_errors)
    net_errors_div_diff = 2 ** np.diff(net_errors_log)
    net_errors_mean = np.mean(net_errors, axis=0)
    net_errors_std = np.std(net_errors, axis=0)
    net_errors_div_diff_mean = np.mean(net_errors_div_diff, axis=0)
    net_errors_div_diff_std = np.std(net_errors_div_diff, axis=0)

    black_box_errors = np.array(black_box_errors)
    black_box_errors_log = np.log2(black_box_errors)
    black_box_errors_div_diff = 2 ** np.diff(black_box_errors_log)
    black_box_errors_mean = np.mean(black_box_errors, axis=0)
    black_box_errors_std = np.std(black_box_errors, axis=0)
    black_box_errors_div_diff_mean = np.mean(black_box_errors_div_diff, axis=0)
    black_box_errors_div_diff_std = np.std(black_box_errors_div_diff, axis=0)

    plt.figure()
    plt.plot(np.arange(len(net_errors_mean), dtype=np.int), net_errors_mean, label='nn')
    plt.plot(np.arange(len(black_box_errors_mean), dtype=np.int), black_box_errors_mean, label='black box')
    plt.xticks(np.arange(len(black_box_errors_mean), step=10))
    plt.xlabel('iteration number')
    plt.ylabel('error l2 norm')
    plt.yscale('log')
    plt.legend()
    plt.savefig('results/test_p.png')

    plt.figure()
    plt.plot(np.arange(len(net_errors_div_diff_mean), dtype=np.int), net_errors_div_diff_mean, label='nn')
    plt.plot(np.arange(len(black_box_errors_div_diff_mean), dtype=np.int), black_box_errors_div_diff_mean,
             label='black box')
    plt.xticks(np.arange(len(black_box_errors_mean), step=10))
    plt.xlabel('iteration number')
    plt.ylabel('error l2 norm')
    plt.legend()
    plt.savefig('results/test_div_diff.png')

    results_file = open("results/results.txt", 'w')
    print(f"network asymptotic error factor: {net_errors_div_diff_mean[-1]:.4f} ± {net_errors_div_diff_std[-1]:.5f}",
          file=results_file)
    print(
        f"black box asymptotic error factor: {black_box_errors_div_diff_mean[-1]:.4f} ± {black_box_errors_div_diff_std[-1]:.5f}",
        file=results_file)
    net_success_rate = sum(net_errors_div_diff[:, -1] < black_box_errors_div_diff[:, -1]) / args.num_test_samples
    print(f"network success rate: {100 * net_success_rate}%",
          file=results_file)
    if args.compute_spectral_radius:
        network_spectral_radii = np.array(network_spectral_radii)
        network_spectral_mean = network_spectral_radii.mean()
        network_spectral_std = network_spectral_radii.std()
        network_frob_norms = np.array(network_frob_norms)
        network_frob_mean = network_frob_norms.mean()
        network_frob_std = network_frob_norms.std()
        black_box_spectral_radii = np.array(black_box_spectral_radii)
        black_box_spectral_mean = black_box_spectral_radii.mean()
        black_box_spectral_std = black_box_spectral_radii.std()
        black_box_frob_norms = np.array(black_box_frob_norms)
        black_box_frob_mean = black_box_frob_norms.mean()
        black_box_frob_std = black_box_frob_norms.std()
        print(f"network spectral radius: {network_spectral_mean:.4f} ± {network_spectral_std:.5f}",
              file=results_file)
        print(f"network frobenius norm: {network_frob_mean:.4f} ± {network_frob_std:.5f}",
              file=results_file)
        print(f"black box spectral radius: {black_box_spectral_mean:.4f} ± {black_box_spectral_std:.5f}",
              file=results_file)
        print(f"black box frobenius norm: {black_box_frob_mean:.4f} ± {black_box_frob_std:.5f}",
              file=results_file)
    results_file.close()
