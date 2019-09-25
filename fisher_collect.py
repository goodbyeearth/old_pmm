# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Train a ConvNet on MNIST using K-FAC.

This library fits a 5-layer ConvNet on MNIST using K-FAC. The model has the
following structure,

- Conv Layer: 5x5 kernel, 16 output channels.
- Max Pool: 3x3 kernel, stride 2.
- Conv Layer: 5x5 kernel, 16 output channels.
- Max Pool: 3x3 kernel, stride 2.
- Linear: 10 output dims.

After 3k~6k steps, this should reach perfect accuracy on the training set.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

from example import mlp
from example import mnist
from tensorflow.contrib.kfac.python.ops import optimizer as opt
from absl import flags
from tensorflow.examples.tutorials.mnist import input_data
import joblib
# import os
#
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


lc = tf.contrib.kfac.layer_collection
oq = tf.contrib.kfac.op_queue
opt = tf.contrib.kfac.optimizer

__all__ = [
    "conv_layer",
    "max_pool_layer",
    "linear_layer",
    "build_model",
    "minimize_loss_single_machine",
    "distributed_grads_only_and_ops_chief_worker",
    "distributed_grads_and_ops_dedicated_workers",
    "train_mnist_single_machine",
    "train_mnist_distributed_sync_replicas",
    "train_mnist_multitower"
]


# Inverse update ops will be run every _INVERT_EVRY iterations.
_INVERT_EVERY = 10


def conv_layer(layer_id, inputs, kernel_size, out_channels):
  """Builds a convolutional layer with ReLU non-linearity.

  Args:
    layer_id: int. Integer ID for this layer's variables.
    inputs: Tensor of shape [num_examples, width, height, in_channels]. Each row
      corresponds to a single example.
    kernel_size: int. Width and height of the convolution kernel. The kernel is
      assumed to be square.
    out_channels: int. Number of output features per pixel.

  Returns:
    preactivations: Tensor of shape [num_examples, width, height, out_channels].
      Values of the layer immediately before the activation function.
    activations: Tensor of shape [num_examples, width, height, out_channels].
      Values of the layer immediately after the activation function.
    params: Tuple of (kernel, bias), parameters for this layer.
  """
  # TODO(b/67004004): Delete this function and rely on tf.layers exclusively.
  layer = tf.layers.Conv2D(
      out_channels,
      kernel_size=[kernel_size, kernel_size],
      kernel_initializer=tf.random_normal_initializer(stddev=0.01),
      padding="VALID",
      name="conv_%d" % layer_id)
  preactivations = layer(inputs)
  activations = tf.nn.relu(preactivations)

  # layer.weights is a list. This converts it a (hashable) tuple.
  return preactivations, activations, (layer.kernel, layer.bias)


def max_pool_layer(layer_id, inputs, kernel_size, stride):
  """Build a max-pooling layer.

  Args:
    layer_id: int. Integer ID for this layer's variables.
    inputs: Tensor of shape [num_examples, width, height, in_channels]. Each row
      corresponds to a single example.
    kernel_size: int. Width and height to pool over per input channel. The
      kernel is assumed to be square.
    stride: int. Step size between pooling operations.

  Returns:
    Tensor of shape [num_examples, width/stride, height/stride, out_channels].
    Result of applying max pooling to 'inputs'.
  """
  # TODO(b/67004004): Delete this function and rely on tf.layers exclusively.
  with tf.variable_scope("pool_%d" % layer_id):
    return tf.nn.max_pool(
        inputs, [1, kernel_size, kernel_size, 1], [1, stride, stride, 1],
        padding="SAME",
        name="pool")


def linear_layer(layer_id, inputs, output_size):
  """Builds the final linear layer for an MNIST classification problem.

  Args:
    layer_id: int. Integer ID for this layer's variables.
    inputs: Tensor of shape [num_examples, width, height, in_channels]. Each row
      corresponds to a single example.
    output_size: int. Number of output dims per example.

  Returns:
    activations: Tensor of shape [num_examples, output_size]. Values of the
      layer immediately after the activation function.
    params: Tuple of (weights, bias), parameters for this layer.
  """
  # TODO(b/67004004): Delete this function and rely on tf.layers exclusively.
  pre, act, params = mlp.fc_layer(layer_id, inputs, output_size)
  return pre, act,params


def build_model(examples, labels, num_labels, layer_collection,device):
  """Builds a ConvNet classification model.

  Args:
    examples: Tensor of shape [num_examples, num_features]. Represents inputs of
      model.
    labels: Tensor of shape [num_examples]. Contains integer IDs to be predicted
      by softmax for each example.
    num_labels: int. Number of distinct values 'labels' can take on.
    layer_collection: LayerCollection instance. Layers will be registered here.

  Returns:
    loss: 0-D Tensor representing loss to be minimized.
    accuracy: 0-D Tensor representing model's accuracy.
  """
  # Build a ConvNet. For each layer with parameters, we'll keep track of the
  # preactivations, activations, weights, and bias.
  tf.logging.info("Building model.")
  pre0, act0, params0 = conv_layer(
      layer_id=0, inputs=examples, kernel_size=3, out_channels=32)

  # act1 = max_pool_layer(layer_id=1, inputs=act0, kernel_size=2, stride=2)

  pre2, act2, params2 = conv_layer(
      layer_id=2, inputs=act0, kernel_size=3, out_channels=64)

  # act3 = max_pool_layer(layer_id=6, inputs=act2, kernel_size=2, stride=2)

  pre4, act4, params4 = conv_layer(
      layer_id=3, inputs=act2, kernel_size=3, out_channels=64)

  flat_act5 = tf.reshape(act4, shape=[-1, int(np.prod(act4.shape[1:4]))])

  pre5, act5,params5 = linear_layer(
      layer_id=4, inputs=flat_act5, output_size=512)

  logits,_,params6 = linear_layer(layer_id=5,inputs=act5,output_size=num_labels)
  loss = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels, logits=logits))
  accuracy = tf.reduce_mean(
      tf.cast(tf.equal(labels, tf.argmax(logits, axis=1)), dtype=tf.float32))

  val_list = list(params0+params2+params4+params5+params6)

  with tf.device("/cpu:0"):
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)

  # Register parameters. K-FAC needs to know about the inputs, outputs, and
  # parameters of each conv/fully connected layer and the logits powering the
  # posterior probability over classes.
  tf.logging.info("Building LayerCollection.")
  layer_collection.register_conv2d(params0, (1, 1, 1, 1), "VALID", examples,
                                   pre0)
  layer_collection.register_conv2d(params2, (1, 1, 1, 1), "VALID", act0, pre2)
  layer_collection.register_conv2d(params4, (1, 1, 1, 1), "VALID", act2, pre4)
  layer_collection.register_fully_connected(params5, flat_act5, pre5)
  layer_collection.register_fully_connected(params6, act5, logits)
  layer_collection.register_categorical_predictive_distribution(
      logits, name="logits")

  g_step = tf.train.get_or_create_global_step()
  optimizer = opt.KfacOptimizer(
      learning_rate=0.0001,
      cov_ema_decay=0.95,
      damping=0.001,
      layer_collection=layer_collection,
      placement_strategy="round_robin",
      cov_devices=[device],
      inv_devices=[device],
      momentum=0.9)

  data = np.load('distillation_data/simple_random_3.npz')
  observation = data['observation'][:6000]

  (cov_update_thunks, inv_update_thunks) = optimizer.make_vars_and_create_op_thunks()
  sess = tf.InteractiveSession()

  def make_update_op(update_thunks):
      update_ops = [thunk() for thunk in update_thunks]
      return update_ops
      # return tf.group(*update_ops)
  cov_update_op = make_update_op(cov_update_thunks)
  # train_op = optimizer.minimize(loss, global_step=g_step)

  sess.run(tf.global_variables_initializer())
  param = joblib.load('initial_parameter/420000')
  for i in range(10):
      sess.run(val_list[i].assign(param[i]))

  F_accum = []
  num_sample = observation.shape[0]
  print(num_sample)
  for i in range(num_sample):
      F = sess.run(cov_update_op, feed_dict={examples:observation[i:i+1]})
      for index in range(len(F)):
          if i == 0:
              F_accum.append(F[index])
          else:
              F_accum[index] += F[index]
  for i in range(len(F_accum)):
      F_accum[i] /= num_sample
  joblib.dump(F_accum,'fisher_matrix_tf/simple_agent_3_random_6000')
  return

def minimize_loss_single_machine(loss,
                                 accuracy,
                                 layer_collection,
                                 device="/gpu:0",
                                 session_config=None):
  # Train with K-FAC.
  g_step = tf.train.get_or_create_global_step()
  optimizer = opt.KfacOptimizer(
      learning_rate=0.0001,
      cov_ema_decay=0.95,
      damping=0.001,
      layer_collection=layer_collection,
      placement_strategy="round_robin",
      cov_devices=[device],
      inv_devices=[device],
      momentum=0.9)

  mnist = input_data.read_data_sets('MNIST_data',reshape=False,one_hot=False)
  sample = mnist.validation.images
  la = mnist.validation.labels

  (cov_update_thunks,inv_update_thunks) = optimizer.make_vars_and_create_op_thunks()
  sess = tf.InteractiveSession()
  def make_update_op(update_thunks):
      update_ops = [thunk() for thunk in update_thunks]
      return update_ops
      # return tf.group(*update_ops)
  sess.run(tf.global_variables_initializer())
  cov_update_op = make_update_op(cov_update_thunks)
  for i in range(200):
      accuracy_ = sess.run(cov_update_op, feed_dict={example:sample[i:i+1],labels:la[i:i+1]})

  # with tf.control_dependencies([cov_update_op]):
  #   inverse_op = tf.cond(
  #       tf.equal(tf.mod(g_step, _INVERT_EVERY), 0),
  #       lambda: make_update_op(inv_update_thunks), tf.no_op)
  #   with tf.control_dependencies([inverse_op]):
  #     with tf.device(device):
  #       train_op = optimizer.minimize(loss, global_step=g_step)

  # tf.logging.info("Starting training.")
  # with tf.train.MonitoredTrainingSession(config=session_config) as sess:
  #   while not sess.should_stop():
  #     # global_step_, loss_, accuracy_, _ = sess.run(
  #     #     [g_step, loss, accuracy, train_op])
  #     accuracy_ = sess.run(cov_update_op)

      # if global_step_ % _INVERT_EVERY == 0:
      #   tf.logging.info("global_step: %d | loss: %f | accuracy: %s",
      #                   global_step_, loss_, accuracy_)

  return accuracy_


def train_mnist_single_machine(data_dir,
                               num_epochs,
                               use_fake_data=False,
                               device="/gpu:0"):
  """Train a ConvNet on MNIST.

  Args:
    data_dir: string. Directory to read MNIST examples from.
    num_epochs: int. Number of passes to make over the training set.
    use_fake_data: bool. If True, generate a synthetic dataset.
    device: string, Either '/cpu:0' or '/gpu:0'. The covaraince and inverse
      update ops are run on this device.

  Returns:
    accuracy of model on the final minibatch of training data.
  """
  # Load a dataset.
  tf.logging.info("Loading MNIST into memory.")
  # examples, labels = mnist.load_mnist(
  #     data_dir,
  #     num_epochs=num_epochs,
  #     batch_size=128,
  #     use_fake_data=use_fake_data,
  #     flatten_images=False)

  # Build a ConvNet.

  layer_collection = lc.LayerCollection()
  examples = tf.placeholder(tf.float32,shape=[None,8,8,19])
  labels = tf.placeholder(tf.int64,shape=[None])

  build_model(
      examples, labels, num_labels=6, layer_collection=layer_collection,device=device)

  # Fit model.
  # return minimize_loss_single_machine(
  #     loss, accuracy, layer_collection, device=device)

def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS = flags.FLAGS
    flags.DEFINE_string("data_dir", "/tmp/mnist", "local mnist dir")
    train_mnist_single_machine(FLAGS.data_dir, num_epochs=200)



if __name__ == "__main__":
  tf.app.run(main=main)