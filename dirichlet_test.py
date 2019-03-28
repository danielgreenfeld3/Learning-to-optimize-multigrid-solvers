import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from utils_dirichletBC import *
from model_dirichletBC import my_model

batch_size = 1
num_training_samples =  2#232*16384
num_test_samples = 1
grid_size = 255

m = my_model(grid_size=grid_size)

with tf.device("gpu:0"):
    pi = tf.constant(np.pi)
    ci = tf.to_complex128(1j)
n_test = 255

checkpoint_dir = ''
checkpoint_perfix = os.path.join(checkpoint_dir, 'ckpt')

with tf.device("gpu:0"):
    lr = tfe.Variable([3.4965356e-05])
    optimizer = tf.train.AdamOptimizer(lr)
root = tf.train.Checkpoint(optimizer=optimizer,model=m, optimizer_step=tf.train.get_or_create_global_step())
root.restore(tf.train.latest_checkpoint(checkpoint_dir))

black_box_results = []
black_box_errors = []
net_results = []
net_errors = []
for iter in range(500):
    print(iter)
    A_stencils_test = two_d_stencil(-1)
    A_matrices_test = compute_A(A_stencils_test, grid_size=grid_size)
    A_stencils_test = tf.convert_to_tensor([A_stencils_test], dtype=tf.double)
    b = np.zeros((grid_size**2,1))
    tmp_ = np.random.normal(size=(grid_size**2))

    tmp = tmp_
    num_iterations = 20
    residuals = []
    errors = []
    cache = []
    for _ in range(num_iterations):
        errors.append(np.linalg.norm(tmp))
        tmp,cache = cycle(m,n_test,A_stencils_test,A_matrices_test,
                            b,tmp,grid_size,0,6,bb=True,cache=cache,w_cycle=False)
    black_box_errors.append(errors)
    black_box_results.append(residuals)
    residuals = []
    errors = []
    cache = []
    tmp = tmp_
    for _ in range(num_iterations):
        errors.append(np.linalg.norm(tmp))
        tmp, cache = cycle(m, n_test, A_stencils_test, A_matrices_test,
                             b, tmp, grid_size, 0, 6, bb=False, cache=cache,w_cycle=False)

net_results = np.array(net_results)
black_box_results = np.array(black_box_results)
print(np.mean(net_results[:,0]))
print(np.std(net_results[:,0]))
print(np.mean(net_results[:,1]))
print(np.std(net_results[:,1]))