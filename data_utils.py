from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import _pickle as pickle
import numpy as np
import tensorflow as tf


def _parse_data(data_path, train_files):
  """Reads CIFAR-10 format data. Always returns NHWC format.

  Returns:
    images: np tensor of size [N, H, W, C]
    labels: np tensor of size [N]
  """
  
  images, labels = [], []
  for file_name in train_files:
    print("Reading:" + file_name)
    full_name = os.path.join(data_path, file_name)
    data = {}
    with open(full_name, 'rb') as finp:
      bdata = pickle.load(finp,  encoding='bytes')
      for k, bli in bdata.items():
          data[k.decode('utf-8')] = bli
      batch_images = data["data"].astype(np.float32) / 255.0
      batch_labels = np.array(data["labels"], dtype=np.int32)
      images.append(batch_images)
      labels.append(batch_labels)
  images = np.concatenate(images, axis=0)
  labels = np.concatenate(labels, axis=0)
  images = np.reshape(images, [-1, 3, 32, 32])
  images = np.transpose(images, [0, 2, 3, 1])

  return images, labels


def read_data(data_path, num_valids=5000):
  """Read the data and perform center normalization.

  Args:
    data_path: path to the data folder.
    num_valids: number of images reserved from the training images to use as
      validation data.

  Returns:
    images, labels: two dicts, each with keys ['train', 'valid', 'test']. They
      contain the images and labels for the corresponding category.
  """

  print("-" * 80)
  print("Reading data")

  images, labels = {}, {}

  train_files = [
    "data_batch_1",
    "data_batch_2",
    "data_batch_3",
    "data_batch_4",
    "data_batch_5",
  ]
  test_file = [
    "test_batch",
  ]
  images["train"], labels["train"] = _parse_data(data_path, train_files)

  images["valid"] = images["train"][-num_valids:]
  labels["valid"] = labels["train"][-num_valids:]

  images["train"] = images["train"][:-num_valids]
  labels["train"] = labels["train"][:-num_valids]

  images["test"], labels["test"] = _parse_data(data_path, test_file)

  print("Prepropcess: [subtract mean], [divide std]")
  mean = np.mean(images["train"], axis=(0, 1, 2), keepdims=True)
  std = np.std(images["train"], axis=(0, 1, 2), keepdims=True)

  print("mean: {}".format(np.reshape(mean * 255.0, [-1])))
  print("std: {}".format(np.reshape(std * 255.0, [-1])))

  images["train"] = (images["train"] - mean) / std
  images["valid"] = (images["valid"] - mean) / std
  images["test"] = (images["test"] - mean) / std

  return images, labels


def _pre_process(img, label):
  """Process image only"""
  img = tf.pad(img, [[4, 4], [4, 4], [0, 0]])
  img = tf.random_crop(img, [32, 32, 3])
  img = tf.image.random_flip_left_right(img)
  return img, label

def create_batch_tf_dataset(images, labels,
                            batch_size=32,
                            n_workers=10,
                            buffer_size=10000,
                            ):
  """
  TODO: you need to fill in this method  
  
  Args:
    images: from read_data()
    labels: from read_data() 
    batch_size: 
    n_workers: paralellization factor to speed up the data consumption 
    buffer_size: memory needed for queueing the cached data 
  
  Returns: 
    a dictionary containing 3 batched tf.data.Dataset objects 
  """
  # train dataset
  train_dataset = tf.data.Dataset.from_tensor_slices()
  # TODO: optionally preprocess and make it loop forever
  batched_train_dataset = None

  # val dataset
  batched_val_dataset = None

  # test dataset
  batched_test_dataset = None

  return {
    "train": batched_train_dataset,
    "valid": batched_val_dataset,
    "test": batched_test_dataset
  }


if __name__ == "__main__":
  # sample test driver here
  # make sure this runnable before moving on
  images, labels = read_data(".")
  dataset_dict = create_batch_tf_dataset(images, labels)

  train_dataset = dataset_dict["valid"]
  iterator = train_dataset.make_initializable_iterator()
  next_element = iterator.get_next()

  import pdb; pdb.set_trace()

  with tf.Session() as sess:
    for iter in range(10):
      sess.run(iterator.initializer)  # reset at each epoch
      while True:
        try:
          imgs, labels = sess.run(next_element)
        except tf.errors.OutOfRangeError:
          break
