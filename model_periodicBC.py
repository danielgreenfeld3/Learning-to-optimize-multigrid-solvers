import numpy as np
import tensorflow as tf


class Pnetwork(tf.keras.Model):
    def __init__(self, grid_size=8, device="/cpu:0"):
        super(Pnetwork, self).__init__()
        self.grid_size = grid_size
        self.device = device
        with tf.device(device):
            width = 100
            self.linear0 = tf.keras.layers.Dense(width,kernel_regularizer=tf.keras.regularizers.l2(1e-7),
                                                 use_bias=False)
            self.num_layers = 100
            for i in range(1, self.num_layers):
                setattr(self, "linear%i" % i, tf.keras.layers.Dense(width, use_bias=False,
                                                                    kernel_regularizer=tf.keras.regularizers.l2(1e-7),
                                                                    kernel_initializer=tf.initializers.truncated_normal(
                                                                        stddev=i ** (-1 / 2) * np.sqrt(2. / width))))
                setattr(self, "bias_1%i" % i, tf.Variable([0.], dtype=tf.float64))
                setattr(self, "linear%i" % (i + 1), tf.keras.layers.Dense(width, use_bias=False,
                                                                          kernel_regularizer=tf.keras.regularizers.l2(
                                                                              1e-7),
                                                                          kernel_initializer=tf.zeros_initializer()))
                setattr(self, "bias_2%i" % i, tf.Variable([0.], dtype=tf.float64))
                setattr(self, "bias_3%i" % i, tf.Variable([0.], dtype=tf.float64))
                setattr(self, "bias_4%i" % i, tf.Variable([0.], dtype=tf.float64))
                setattr(self, "multiplier_%i" % i, tf.Variable([1.], dtype=tf.float64))

            self.output_layer = tf.keras.layers.Dense(4, use_bias=True, kernel_regularizer=tf.keras.regularizers.l2(1e-5))
            self.new_output = tf.Variable(0.5*tf.random_normal(shape=[2*2*2*8], dtype=tf.float64), dtype=tf.float64)

    def call(self, inputs, black_box=False, index=None, pos=-1., phase='Training'):

        with tf.device(self.device):
            batch_size = inputs.shape[0]
            right_contributions_input = tf.gather(params=inputs,
                                                  indices=[i for i in range(1, self.grid_size, 2)], axis=1)
            right_contributions_input = tf.gather(params=right_contributions_input,
                                                  indices=[i for i in range(0, self.grid_size, 2)], axis=2)
            idx = [(i-1) % self.grid_size for i in range(0, self.grid_size, 2)]
            left_contributions_input = tf.gather(params=inputs, indices=idx, axis=1)
            left_contributions_input = tf.gather(params=left_contributions_input,
                                                 indices=[i for i in range(0, self.grid_size, 2)], axis=2)
            left_contributions_input = tf.reshape(tensor=left_contributions_input,
                                                  shape=(-1, self.grid_size//2, self.grid_size//2, 3, 3))

            up_contributions_input = tf.gather(params=inputs, indices=[i for i in range(0, self.grid_size, 2)], axis=1)
            up_contributions_input = tf.gather(params=up_contributions_input,
                                               indices=[i for i in range(1, self.grid_size, 2)], axis=2)
            up_contributions_input = tf.reshape(tensor=up_contributions_input,
                                                shape=(-1, self.grid_size//2, self.grid_size//2, 3, 3))

            down_contributions_input = tf.gather(params=inputs,
                                                 indices=[i for i in range(0,self.grid_size,2)], axis=1)
            down_contributions_input = tf.gather(params=down_contributions_input, indices=idx, axis=2)
            down_contributions_input = tf.reshape(tensor=down_contributions_input,
                                                  shape=(-1,self.grid_size//2,self.grid_size//2,3,3))
            #
            center_contributions_input = tf.gather(params=inputs,
                                                   indices=[i for i in range(0, self.grid_size, 2)], axis=1)
            center_contributions_input = tf.gather(params=center_contributions_input, indices=[i for i in range(0, self.grid_size, 2)], axis=2)
            center_contributions_input = tf.reshape(tensor=center_contributions_input,
                                                    shape=(-1, self.grid_size // 2, self.grid_size // 2, 3, 3))

            inputs_combined = tf.concat([right_contributions_input, left_contributions_input,
                                         up_contributions_input, down_contributions_input,
                                         center_contributions_input], 0)

            flattended = tf.reshape(inputs_combined, (-1, 9))

            temp = (self.grid_size//2)**2

            # bug then augmented with doubled grid size
            flattended = tf.concat([flattended[:batch_size*temp],
                                    flattended[temp*batch_size:temp*2*batch_size],
                                    flattended[temp*2*batch_size:temp*3*batch_size],
                                    flattended[temp*3*batch_size:temp*4*batch_size],
                                    flattended[temp*4*batch_size:]],-1)

            x = self.linear0(flattended)
            x = tf.nn.relu(x)
            for i in range(1, self.num_layers, 2):
                x1 = getattr(self, "bias_1%i" % i) + x
                x1 = getattr(self, "linear%i" % i)(x1)
                x1 = x1 + getattr(self, "bias_2%i" % i) + x1
                x1 = tf.nn.relu(x1)
                x1 = x1 + getattr(self, "bias_3%i" % i) + x1
                x1 = getattr(self, "linear%i" % (i + 1))(x1)
                x1 = tf.multiply(x1, getattr(self, "multiplier_%i" % i))
                x = x + x1
                x = x + getattr(self, "bias_4%i" % i)
                x = tf.nn.relu(x)

            x = self.output_layer(x)

            if index is not None:
                indices = tf.constant([[index]])
                updates = [tf.to_double(pos)]
                shape = tf.constant([2*2*2*8])
                scatter = tf.scatter_nd(indices,updates,shape)
                x = self.new_output+tf.reshape(scatter,(-1,2,2,8))
                ld_contribution = x[:, :, :, 0]
                left_contributions_output = x[:, :, :, 1]
                lu_contribution = x[:, :, :, 2]
                down_contributions_output = x[:, :, :, 3]
                up_contributions_output = x[:, :, :, 4]
                ones = tf.ones_like(up_contributions_output)
                right_contributions_output = x[:, :, :, 6]
                rd_contribution = x[:, :, :, 5]
                ru_contribution = x[:, :, :, 7]
                first_row = tf.concat(
                    [tf.expand_dims(ld_contribution, -1), tf.expand_dims(left_contributions_output, -1),
                     tf.expand_dims(lu_contribution, -1)], -1)
                second_row = tf.concat([tf.expand_dims(down_contributions_output, -1),
                                        tf.expand_dims(ones, -1), tf.expand_dims(up_contributions_output, -1)], -1)
                third_row = tf.concat(
                    [tf.expand_dims(rd_contribution, -1), tf.expand_dims(right_contributions_output, -1),
                     tf.expand_dims(ru_contribution, -1)], -1)

                output = tf.stack([first_row, second_row, third_row], 0)
                output = tf.transpose(output, (1, 2, 3, 0, 4))
                if not black_box:
                    return tf.to_complex128(output)
            else:
                x = tf.reshape(x, (-1, self.grid_size//2, self.grid_size//2,4))
            if black_box:
                up_contributions_output = tf.gather(inputs,[i for i in range(0,self.grid_size,2)],axis=1)
                up_contributions_output = tf.gather(up_contributions_output,
                                                    [i for i in range(1,self.grid_size,2)], axis=2)
                up_contributions_output = -tf.reduce_sum(up_contributions_output[:,:,:,:,0],axis=-1)/tf.reduce_sum(up_contributions_output[:,:,:,:,1],axis=-1)

                left_contributions_output = tf.gather(inputs, idx, axis=1)
                left_contributions_output = tf.gather(left_contributions_output,
                                                      [i for i in range(0,self.grid_size,2)], axis=2)
                left_contributions_output = -tf.reduce_sum(left_contributions_output[:, :, :, 2, :],
                                                            axis=-1) / tf.reduce_sum(
                    left_contributions_output[:, :, :, 1, :], axis=-1)

                right_contributions_output = tf.gather(inputs, [i for i in range(1,self.grid_size,2)], axis=1)
                right_contributions_output = tf.gather(right_contributions_output, [i for i in range(0,self.grid_size,2)], axis=2)
                right_contributions_output = -tf.reduce_sum(right_contributions_output[:, :, :, 0, :],
                                                           axis=-1) / tf.reduce_sum(
                    right_contributions_output[:, :, :, 1, :], axis=-1)
                down_contributions_output = tf.gather(inputs, [i for i in range(0,self.grid_size,2)], axis=1)
                down_contributions_output = tf.gather(down_contributions_output, idx, axis=2)
                down_contributions_output = -tf.reduce_sum(down_contributions_output[:, :, :, :, 2],
                                                            axis=-1) / tf.reduce_sum(
                    down_contributions_output[:, :, :, :, 1], axis=-1)
            else:
                jm1 = [(i - 1) % (self.grid_size // 2) for i in range(self.grid_size // 2)]
                jp1 = [(i + 1) % (self.grid_size // 2) for i in range(self.grid_size // 2)]
                right_contributions_output = x[:,:,:,0]/(tf.gather(x[:, :, :, 1],jp1,axis=1)+x[:,:,:,0])
                left_contributions_output = x[:,:,:,1]/(x[:,:,:,1]+tf.gather(x[:, :, :, 0],jm1,axis=1))
                up_contributions_output = x[:,:,:,2]/(x[:,:,:,2]+tf.gather(x[:, :, :, 3],jp1,axis=2))
                down_contributions_output = x[:,:,:,3]/(tf.gather(x[:, :, :, 2],jm1,axis=2)+x[:,:,:,3])
            ones = tf.ones_like(down_contributions_output)

            #based on rule 2 given rule 1:
            up_right_contribution = tf.gather(inputs,[i for i in range(1,self.grid_size,2)],axis=1)
            up_right_contribution = tf.gather(up_right_contribution, [i for i in range(1,self.grid_size,2)], axis=2)
            up_right_contribution = up_right_contribution [:,:,:,0,1]
            right_up_contirbution = tf.gather(inputs, [i for i in range(1,self.grid_size,2)], axis=1)
            right_up_contirbution = tf.gather(right_up_contirbution, [i for i in range(1,self.grid_size,2)], axis=2)
            right_up_contirbution_additional_term = right_up_contirbution[:, :, :, 0, 0]
            right_up_contirbution = right_up_contirbution[:,:,:,1,0]
            ru_center_ = tf.gather(inputs, [i for i in range(1,self.grid_size,2)], axis=1)
            ru_center_ = tf.gather(ru_center_, [i for i in range(1,self.grid_size,2)], axis=2)
            ru_center_ = ru_center_[:,:,:,1,1]
            ru_contribution = -tf.expand_dims((right_up_contirbution_additional_term+
                                               tf.multiply(right_up_contirbution,right_contributions_output) +\
                              tf.multiply(up_right_contribution,up_contributions_output))/ru_center_, -1)

            up_left_contribution = tf.gather(inputs, idx, axis=1)
            up_left_contribution = tf.gather(up_left_contribution, [i for i in range(1,self.grid_size,2)], axis=2)
            up_left_contribution = up_left_contribution[:, :, :, 2, 1]
            left_up_contirbution = tf.gather(inputs, idx, axis=1)
            left_up_contirbution = tf.gather(left_up_contirbution, [i for i in range(1,self.grid_size,2)], axis=2)
            left_up_contirbution_addtional_term = left_up_contirbution[:, :, :, 2, 0]
            left_up_contirbution = left_up_contirbution[:, :, :, 1, 0]
            lu_center_ = tf.gather(inputs, idx, axis=1)
            lu_center_ = tf.gather(lu_center_, [i for i in range(1,self.grid_size,2)], axis=2)
            lu_center_ = lu_center_[:, :, :, 1, 1]
            lu_contribution = -tf.expand_dims((left_up_contirbution_addtional_term+
                                               tf.multiply(up_left_contribution , up_contributions_output) + \
                               tf.multiply(left_up_contirbution , left_contributions_output)) / lu_center_, -1)

            down_left_contribution = tf.gather(inputs, idx, axis=1)
            down_left_contribution = tf.gather(down_left_contribution, idx, axis=2)
            down_left_contribution = down_left_contribution[:, :, :, 2, 1]
            left_down_contirbution = tf.gather(inputs, idx, axis=1)
            left_down_contirbution = tf.gather(left_down_contirbution, idx, axis=2)
            left_down_contirbution_additional_term = left_down_contirbution[:, :, :, 2, 2]
            left_down_contirbution = left_down_contirbution[:, :, :, 1, 2]
            ld_center_ = tf.gather(inputs, idx, axis=1)
            ld_center_ = tf.gather(ld_center_, idx, axis=2)
            ld_center_ = ld_center_[:, :, :, 1, 1]
            ld_contribution = -tf.expand_dims((left_down_contirbution_additional_term+
                                               tf.multiply(down_left_contribution , down_contributions_output) + \
                               tf.multiply(left_down_contirbution , left_contributions_output)) / ld_center_,-1)

            down_right_contribution = tf.gather(inputs, [i for i in range(1,self.grid_size,2)], axis=1)
            down_right_contribution = tf.gather(down_right_contribution, idx, axis=2)
            down_right_contribution = down_right_contribution[:, :, :, 0, 1]
            right_down_contirbution = tf.gather(inputs, [i for i in range(1,self.grid_size,2)], axis=1)
            right_down_contirbution = tf.gather(right_down_contirbution, idx, axis=2)
            right_down_contirbution_addtional_term = right_down_contirbution[:, :, :, 0, 2]
            right_down_contirbution = right_down_contirbution[:, :, :, 1, 2]
            rd_center_ = tf.gather(inputs, [i for i in range(1,self.grid_size,2)], axis=1)
            rd_center_ = tf.gather(rd_center_, idx, axis=2)
            rd_center_ = rd_center_[:, :, :, 1, 1]
            rd_contribution = -tf.expand_dims((right_down_contirbution_addtional_term+tf.multiply(down_right_contribution , down_contributions_output) + \
                               tf.multiply(right_down_contirbution , right_contributions_output)) / rd_center_,-1)

            first_row = tf.concat([ld_contribution, tf.expand_dims(left_contributions_output,-1),
                                   lu_contribution], -1)
            second_row = tf.concat([tf.expand_dims(down_contributions_output,-1),
                                    tf.expand_dims(ones, -1), tf.expand_dims(up_contributions_output, -1)], -1)
            third_row = tf.concat([rd_contribution, tf.expand_dims(right_contributions_output, -1),
                                   ru_contribution], -1)

            output = tf.stack([first_row, second_row, third_row], 0)
            output = tf.transpose(output, (1, 2, 3, 0, 4))

            return tf.to_complex128(output)