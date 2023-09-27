import numpy as np
import tensorflow as tf


def dense(layer, n_units):
  layer = tf.compat.v1.layers.dense(
    layer, n_units, activation=None,
    kernel_regularizer=tf.keras.regularizers.l2(0.5 * (1.0)),
    kernel_initializer=tf.compat.v1.initializers.truncated_normal(stddev=0.01)
  )
  return layer


def dense_bn(layer, n_units, training):
  layer = dense(layer, n_units)
  layer = tf.compat.v1.layers.batch_normalization(layer, training=training)
  return layer


def iknet(xyz, depth, width, training):
  N = xyz.get_shape().as_list()[0]
  layer = tf.reshape(xyz, [N, -1])
  for _ in range(depth):
    layer = dense_bn(layer, width, training)
    layer = tf.nn.sigmoid(layer)
  theta_raw = dense(layer, 21 * 4)
  theta_raw = tf.reshape(theta_raw, [-1, 21, 4])
  eps = np.finfo(np.float32).eps
  norm = tf.maximum(tf.norm(tensor=theta_raw, axis=-1, keepdims=True), eps)
  theta_pos = theta_raw / norm
  theta_neg = theta_pos * -1
  theta = tf.compat.v1.where(
    tf.tile(theta_pos[:, :, 0:1] > 0, [1, 1, 4]), theta_pos, theta_neg
  )
  return theta, norm
