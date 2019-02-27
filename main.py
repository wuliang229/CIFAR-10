from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import sys

import tensorflow as tf

from data_utils import read_data
from models import create_tf_ops
from utils import DEFINE_boolean
from utils import DEFINE_float
from utils import DEFINE_integer
from utils import DEFINE_string
from utils import print_user_flags

flags = tf.app.flags
FLAGS = flags.FLAGS

DEFINE_boolean("reset_output_dir", False, "Delete output_dir if exists.")
DEFINE_string("data_path", "", "Path to CIFAR-10 data")
DEFINE_string("output_dir", "output", "Path to log folder")
DEFINE_string("model_name", "conv",
              "Name of the method. [naive|mlp|conv]")
DEFINE_integer("train_steps", 10000, "How many steps to train in total")
DEFINE_integer("log_every", 1000, "How many steps to log")
DEFINE_integer("n_classes", 10, "Number of classes")
DEFINE_integer("batch_size", 32, "Batch size")
DEFINE_float("init_lr", 1e-2, "Init learning rate")


def get_ops(images, labels):
  """Build the computational graph"""
  print("-" * 80)
  print("Creating a '{0}' model".format(FLAGS.model_name))

  ops = create_tf_ops(images, labels,
                      model_type=FLAGS.model_name,
                      n_outputs=FLAGS.n_classes,
                      init_lr=FLAGS.init_lr,
                      batch_size=FLAGS.batch_size)

  assert "global_step" in ops
  assert "train_op" in ops
  assert "train_loss" in ops
  assert "val_preds" in ops
  assert "test_preds" in ops
  assert "val_labels" in ops
  assert "test_labels" in ops
  assert "train_iterator" in ops
  assert "val_iterator" in ops
  assert "test_iterator" in ops

  return ops


def main(_):
  print("-" * 80)
  if not os.path.isdir(FLAGS.output_dir):
    print("Path {0} does not exist. Creating.".format(FLAGS.output_dir))
    os.makedirs(FLAGS.output_dir)
  elif FLAGS.reset_output_dir:
    print("Path {0} exists. Remove and remake.".format(FLAGS.output_dir))
    shutil.rmtree(FLAGS.output_dir)
    os.makedirs(FLAGS.output_dir)

  print_user_flags()

  # data
  images, labels = read_data(FLAGS.data_path)

  # computational graph
  g = tf.Graph()
  with g.as_default():
    ops = get_ops(images, labels)

    print("-" * 80)
    print("Starting session")
    config = tf.ConfigProto(allow_soft_placement=True)

    # hook up with a session to train
    with tf.train.SingularMonitoredSession(
        config=config, checkpoint_dir=FLAGS.output_dir) as sess:

      # training loop
      print("-" * 80)
      print("Starting training")
      sess.run(ops["train_iterator"])  # init dataset iterator

      # train with number of steps (batches) defined above
      for step in range(1, FLAGS.train_steps + 1):
        sess.run(ops["train_op"])

        # report periodically
        if step % FLAGS.log_every == 0:
          get_eval_accuracy(ops, sess, step, "val")

      print("-" * 80)
      print("Training done. Eval on TEST set")
      get_eval_accuracy(ops, sess, step, "test")


def get_eval_accuracy(ops, sess, step, name="val"):
  """
  TODO: run the dataset initializer for 2 cases: val or test 

  Then draw all possible batches to the end of that dataset 
  
  For each batch, compare the preds vs. labels (both taken from `ops`. Then 
  we can finally calculate accuracies including top1  and top5   
  
  """
  if name == "val":
    sess.run(ops["val_iterator"])
    preds_ops = ops["valid_preds"]
    top5_ops = ops["top5_val_preds"]
    labels = ops["val_labels"]
  else:
    sess.run(ops["test_iterator"])
    preds_ops = ops["test_preds"]
    top5_ops = ops["top5_test_preds"]
    labels = ops["test_labels"]


  # TODO: your code here
  n_samples = 0
  top1_acc = 0.0
  top5_acc = 0.0
  # -------------

  log_string = ""
  log_string += "step={0:<6d}".format(step)
  log_string += " top1 acc={0:.3f} top1 acc={0:.3f} against {1:<3d} " \
                "samples".format(top1_acc, top5_acc, n_samples)
  print(log_string)
  sys.stdout.flush()


if __name__ == "__main__":
  tf.app.run()
