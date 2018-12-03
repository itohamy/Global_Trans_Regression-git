
import tensorflow as tf


def expm(params_matrix,batch_size):
    # Take the matrix exponentioal of the affine map, inorder to get an affine-defiomorphism map.
    exp_params = tf.reshape(params_matrix,[-1,2,3])
    # append a row of 0,0,0 before computing the exponent:
    initial = tf.zeros_like(tf.slice(exp_params,[0,0,0],[-1,1,3]))
    initial = tf.cast(initial,tf.float32)
    exp_params = tf.concat([exp_params,initial],1)
    return matrix_expnential(exp_params,batch_size)  #if we want to remove the exp, return this: -params_matrix


def matrix_expnential(matrices,batch_size):
    matrices = tf.cast(matrices,tf.float32)
    x_unpacked = tf.unstack(matrices,num=batch_size)  # defaults to axis 0, returns a list of tensors
    processed = []  # this will be the list of processed tensors
    for t in x_unpacked:
        t = tf.cast(t,tf.float64)
        result_tensor = tf.linalg.expm(t)
        result_tensor = tf.slice(result_tensor,[0,0],[2,3])
        processed.append(result_tensor)

    #output = tf.concat(tf.cast(processed, tf.float32), 0)
    return tf.reshape(processed,[-1,6])
