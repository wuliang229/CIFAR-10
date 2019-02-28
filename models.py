import tensorflow as tf

from data_utils import create_batch_tf_dataset

SEED = 12345678


def _inference(images,
               model_type='conv',
               is_train=True,
               n_outputs=10):
  if model_type == 'naive':
    return _naive_inference(images, is_train=is_train, n_outputs=n_outputs)
  if model_type == 'conv':
    return _conv_inference(images, is_train=is_train, n_outputs=n_outputs)
  if model_type == 'mlp':
    return _mlp_inference(images, is_train=is_train, n_outputs=n_outputs)


def _naive_inference(images,
                     is_train=True,
                     n_outputs=10):
  """Naive model using only 1 FC layer, no dropout. For reference purpose.
    Validation and Test accuracy should be around 26%.
     
    Use this to complete the pipeline first, then after obtaining a similar 
    accuracy, go ahead and implement MLP and CNN by filling up _inference() 
  """
  H, W, C = (images.get_shape()[1].value,
             images.get_shape()[2].value,
             images.get_shape()[3].value)

  # create model parameters same for train, test, val
  with tf.variable_scope("naive", reuse=tf.AUTO_REUSE):
    w_soft = tf.get_variable("w", [H * W * C, n_outputs])

  logits = tf.matmul(images, w_soft)

  return logits


def _conv_inference(images,
                    is_train=True,
                    n_outputs=10):
  """
  From a batch of input images (B x H W C), return a logits (vector 
  of n_classes) 
  
  - Build several stacked convolutional layers (2D) 
  - Place pooling layers properly in between 
  - If is_train: then activate dropout appropriately  
  
  Note: any usage of tf.keras or tf.contrib.slim prebuilt layers is invalid. 
  """
  H, W, C = (images.get_shape()[1].value, 
             images.get_shape()[2].value, 
             images.get_shape()[3].value)
  kernel_sizes = [3]
  num_channels = [5]

  x = images
  for layer_id, (k_size, next_c) in enumerate(zip(kernel_sizes, num_channels)):
    curr_c = x.get_shape()[-1].value
    with tf.variable_scope("layer_{}".format(layer_id), reuse = tf.AUTO_REUSE):
      w = tf.get_variable("w", [k_size, k_size, curr_c, next_c])
    x = tf.nn.conv2d(x, w, padding = "SAME", strides = [1, 1, 1, 1])
    x = tf.nn.relu(x)

  x = tf.reshape(x, [-1, H * W * 5])
  curr_c = x.get_shape()[-1].value
  with tf.variable_scope("logits", reuse=tf.AUTO_REUSE):
    w = tf.get_variable("w", [curr_c, n_outputs])
    logits = tf.matmul(x, w)
  return logits


def _mlp_inference(images,
                   is_train=True,
                   n_outputs=10):
  """
  Build a simple MLP with at least 3 layers to consume inputs and return a 
  logits vector of n_outputs classes.  
  
  Tip: after matmul, activate it with sigmoid, relu, etc.  
  
  if is_train: then apply drop-out appropriately 
  
  Note: any usage of tf.keras or tf.contrib.slim prebuilt layers is invalid. 
  """

  H, W, C = (images.get_shape()[1].value, 
             images.get_shape()[2].value, 
             images.get_shape()[3].value)
  dims = [256, 512, 128]

  x = tf.reshape(images, [-1, H * W * C]) # Flatten
  for layer_id, next_dim in enumerate(dims):
    curr_dim = x.get_shape()[-1].value
    with tf.variable_scope("layer_{}".format(layer_id), reuse=tf.AUTO_REUSE):
      w = tf.get_variable("w", [curr_dim, next_dim])
    x = tf.matmul(x, w)
    x = tf.nn.relu(x)
  curr_dim = x.get_shape()[-1].value
  with tf.variable_scope("logits", reuse=tf.AUTO_REUSE):
    w = tf.get_variable("w", [curr_dim, n_outputs])
    logits = tf.matmul(x, w)
  return logits


def create_tf_ops(images, labels,
                  model_type='conv',
                  n_outputs=10,
                  init_lr=0.001,
                  l2_reg=1e-3,
                  batch_size=32):
  """ Create and finalize a TF graph including ops """
  dataset_dict = create_batch_tf_dataset(images, labels,
                                         batch_size=batch_size)
  train_dataset = dataset_dict["train"]
  train_iterator = train_dataset.make_initializable_iterator()
  train_imgs, train_labels = train_iterator.get_next()

  val_dataset = dataset_dict["valid"]
  val_iterator = val_dataset.make_initializable_iterator()
  val_imgs, val_labels = val_iterator.get_next()

  test_dataset = dataset_dict["test"]
  test_iterator = test_dataset.make_initializable_iterator()
  test_imgs, test_labels = test_iterator.get_next()

  # those 3 graphs shared weights where appropriate
  train_logits = _inference(train_imgs,
                            model_type=model_type,
                            is_train=True,
                            n_outputs=n_outputs)
  val_logits = _inference(val_imgs,
                          model_type=model_type,
                          is_train=False,
                          n_outputs=n_outputs)
  test_logits = _inference(test_imgs,
                           model_type=model_type,
                           is_train=False,
                           n_outputs=n_outputs)

  global_step = tf.Variable(0, dtype=tf.int32, trainable=False,
                            name="global_step")

  # loss function
  xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=train_logits, labels=train_labels)
  train_loss = tf.reduce_mean(xentropy)
  l2_loss = tf.losses.get_regularization_loss()
  train_loss += l2_reg * l2_loss

  # optimizer
  lr = tf.train.exponential_decay(init_lr, global_step * 64,
                                  50000, 0.98, staircase=True)
  optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)

  # train
  train_op = optimizer.minimize(train_loss, global_step=global_step)

  # predictions
  train_preds = tf.to_int32(tf.argmax(train_logits, axis=1))
  val_preds = tf.to_int32(tf.argmax(val_logits, axis=1))
  test_preds = tf.to_int32(tf.argmax(test_logits, axis=1))

  # top 5 predictions
  top5_val_preds = tf.nn.top_k(val_logits, k=5)
  top5_test_preds = tf.nn.top_k(test_logits, k=5)

  # put everything into an ops dict
  ops = {
    "global_step": global_step,
    "train_op": train_op,
    "train_loss": train_loss,
    "train_preds": train_preds,
    "val_preds": val_preds,
    "test_preds": test_preds,
    "train_labels": train_labels,
    "val_labels": val_labels,
    "test_labels": test_labels,
    "top5_val_preds": top5_val_preds,
    "top5_test_preds": top5_test_preds,
    "train_iterator": train_iterator.initializer,
    "val_iterator": val_iterator.initializer,
    "test_iterator": test_iterator.initializer
  }
  return ops
