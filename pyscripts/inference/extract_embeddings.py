"""Training script for multi-gpu training PSPNet/ResNet-101 with SegSort."""

from __future__ import print_function

import argparse
import math
import os
import time
import utils.general

import network.multigpu.layers as nn_mgpu
import network.segsort.train_utils as train_utils
import numpy as np
import tensorflow as tf

import torchvision.transforms as transforms

from seg_models.models.pspnet_mgpu import pspnet_resnet101 as model
from seg_models.image_reader import SegSortImageReader
from seg_models.imagenet_reader import ImageNetReader
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib as mpl
import pdb 

def get_arguments():
  """Parse all the arguments provided from the CLI.
    
  Returns:
    A list of parsed arguments.
  """
  parser = argparse.ArgumentParser(description='Semantic Segmentation')
  # Data parameters.
  parser.add_argument('--batch_size', type=int, default=1,
                      help='Number of images in one step.')
  parser.add_argument('--data_dir', type=str, default='',
                      help='/path/to/dataset/.')
  parser.add_argument('--input_size', type=str, default='336,336',
                      help='Comma-separated string with H and W of image.')
  parser.add_argument('--random_seed', type=int, default=1234,
                      help='Random seed to have reproducible results.')
  parser.add_argument('--num_gpu', type=int, default=2,
                      help='Number of gpus for training.')
  # Training paramters.
  parser.add_argument('--is_training', action='store_true',
                      help='Whether to updates weights.')
  parser.add_argument('--use_global_status', action='store_true',
                      help='Whether to updates moving mean and variance.')
  
  
  parser.add_argument('--num_classes', type=int, default=1000,
                      help='Number of classes to predict.')
  parser.add_argument('--num_loading_workers', type=int, default=10,
                      help='Number of workers to load imagenet.')
  
  # parser.add_argument('--random_mirror', action='store_true',
  #                     help='Whether to randomly mirror the inputs.')
  # parser.add_argument('--random_crop', action='store_true',
  #                     help='Whether to randomly crop the inputs.')
  # parser.add_argument('--random_scale', action='store_true',
  #                     help='Whether to randomly scale the inputs.')
  # SegSort parameters.
  parser.add_argument('--embedding_dim', type=int, default=32,
                      help='Dimension of the feature embeddings.')
  
  # Misc paramters.
  parser.add_argument('--restore_from', type=str, default='',
                      help='Where restore model parameters from.')
  parser.add_argument('--save_pred_every', type=int, default=10000,
                      help='Save summaries and checkpoint every often.')
  parser.add_argument('--update_tb_every', type=int, default=20,
                      help='Update summaries every often.')
  parser.add_argument('--snapshot_dir', type=str, default='',
                      help='Where to save snapshots.')
  parser.add_argument('--save_dir', type=str, default='',
                      help='Where to save numpy embeddings.')
  parser.add_argument('--not_restore_classifier', action='store_true',
                      help='Whether to not restore classifier layers.')

  return parser.parse_args()


def save(saver, sess, logdir, step):
  """Save the trained weights.
   
  Args:
    saver: TensorFlow Saver object.
    sess: TensorFlow session.
    logdir: path to the snapshots directory.
    step: current training step.
  """
  model_name = 'model.ckpt'
  checkpoint_path = os.path.join(logdir, model_name)
    
  if not os.path.exists(logdir):
    os.makedirs(logdir)
  saver.save(sess, checkpoint_path, global_step=step)
  print('The checkpoint has been created.')


def load(saver, sess, ckpt_path):
  """Load the trained weights.
    
  Args:
    saver: TensorFlow Saver object.
    sess: TensorFlow session.
    ckpt_path: path to checkpoint file with parameters.
  """ 

  saver.restore(sess, "./" + ckpt_path)
  print('Restored model parameters from {}'.format(ckpt_path))

def img_loader_sanity(reader):
  img_np, labels_truth = reader.dequeue()
  # pdb.set_trace()
  print(np.max(img_np))
  print(np.min(img_np))
  print(img_np.dtype)
  print(img_np.shape)
  print(labels_truth)
  mean=np.array([122.675, 116.669, 104.008])
  img_np += mean
  print(np.max(img_np))
  print(np.min(img_np))
  img_np /= 255 # for mpl.imsave
  mpl.image.imsave('sanity.jpg', img_np[0])

curr_class_name = ""
curr_idx = 0
def save_numpy_to_dir(save_dir, numpy_list, label_list, idx_to_class, last_batch_dim = None):
    # print(len(numpy_list))
    # print(type(numpy_list[0]))
    # print(numpy_list[0].shape)
    # print(idx_to_class)
    global curr_class_name
    global curr_idx

    #store prev class name
    #if different, create 
    for idx, label in enumerate(label_list):
      if last_batch_dim is not None and idx >= last_batch_dim:
        break

      class_name = idx_to_class[label]

      save_folder = os.path.join(save_dir, curr_class_name)
      if class_name != curr_class_name:
        curr_class_name = class_name
        save_folder = os.path.join(save_dir, curr_class_name)
        #make directory
        if not os.path.exists(save_folder):
          os.makedirs(save_folder)
        curr_idx = 0

      #save numpy at idx to save_dir + curr_class_name
      save_npy = os.path.join(save_folder, "{}.npy".format(curr_idx))
      with open(save_npy, 'wb') as f:
        np.save(f, numpy_list[idx])

      curr_idx += 1
  
def handle_last_batch(img_np, labels_truth, batch_size):
  last_batch_dim = img_np.shape[0]
  tmp = list(img_np.shape)
  tmp[0] = batch_size - last_batch_dim
  padding = np.zeros(tmp)
  padding_l = np.zeros(tmp[0])
  img_np = np.concatenate((img_np, padding), axis=0)
  labels_truth = np.concatenate((labels_truth, padding_l), axis=0)

  return img_np, labels_truth, last_batch_dim

def main():
  print("IMG_NET extracting embeddings")

  """Create the model and start the training."""

  # Read CL arguments and snapshot the arguments into text file.
  args = get_arguments()
  utils.general.snapshot_arg(args)
  global curr_class_name
    
  # The segmentation network is stride 8 by default.
  h, w = map(int, args.input_size.split(','))
  input_size = (h, w)
  innet_size = (int(math.ceil(h/8)), int(math.ceil(w/8)))
    
  # Initialize the random seed.
  tf.set_random_seed(args.random_seed)
    
  # Create queue coordinator.
  coord = tf.train.Coordinator()

  # current step
  step_ph = tf.placeholder(dtype=tf.float32, shape=())

  reader = ImageNetReader(os.path.join(args.data_dir, "train"), args.batch_size, 
    h, args.num_loading_workers, False)

  #num batches: 20019 * 64 = 1281216 (close enough, batch is overest.)

  #set up input
  image_batch = tf.placeholder(tf.float32, [args.batch_size, w, h, 3])
  labels_batch = tf.placeholder(tf.int32, [args.batch_size])

  # Allocated data evenly to each gpu, because batch-size is only 1 -- nvm
  images_mgpu = nn_mgpu.split(image_batch, args.num_gpu) #last gpu is good

  # Create network and output predictions.
  outputs_mgpu = model(images_mgpu, #calls pspnet_resnet101
                       args.embedding_dim,
                       False,
                       args.use_global_status)

  # Grab variable names which should be restored from checkpoints.
  # tw: double check
  restore_var = [
    v for v in tf.global_variables() if 'crop_image_batch' not in v.name]

  # Collect embedding from each gpu.
  with tf.device('/gpu:{:d}'.format(args.num_gpu-1)): #qq: last gpu used to compute loss?
    embedding_list = [outputs[0] for outputs in outputs_mgpu]
    embedding = tf.concat(embedding_list, axis=0) # [batch]x input/8 x input/8 x[emb_size]

  #tw: check how cpc does linear classifier

  #tw: check original training script

  # Set up tf session and initialize variables. 
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  init = tf.global_variables_initializer()
    
  sess.run(init)
  sess.run(tf.local_variables_initializer())
    
  # Load weights.
  loader = tf.train.Saver(var_list=restore_var)
  if args.restore_from is not None:
    print("Loading restore:", args.restore_from)
    load(loader, sess, args.restore_from)

  # Start queue threads.
  threads = tf.train.start_queue_runners(coord=coord, sess=sess)

  # Iterate over training steps.
  pbar = tqdm(range(reader.num_batches))
  print(reader.num_batches)
  #num supposed images: 1281167

  train_save_dir = os.path.join(args.save_dir, args.embedding_dim, "train")
  curr_class_name_tmp = curr_class_name
  for step in pbar:
    start_time = time.time()

    img_np, labels_truth = reader.dequeue()

    #Handle last batch case
    last_batch_dim = None
    if img_np.shape[0] < args.batch_size:
      #last batch
      img_np, labels_truth, last_batch_dim = handle_last_batch(img_np, labels_truth, args.batch_size)
      

    timeA = time.time() - start_time
    start_time = time.time()

    emb_list = sess.run(embedding, feed_dict={image_batch: img_np})

    timeB = time.time() - start_time
    start_time = time.time()

    save_numpy_to_dir(train_save_dir, emb_list, 
      labels_truth, reader.get_idx_to_class(), last_batch_dim)

    timeC = time.time() - start_time
    start_time = time.time()

    if curr_class_name_tmp != curr_class_name:
      curr_class_name_tmp = curr_class_name
      print(curr_class_name)

    print(timeA, timeB, timeC)

    duration = time.time() - start_time
    # desc = 'loss = {:.3f}, lr = {:.6f}'.format(step_loss, lr)
    # pbar.set_description(desc)
  
  coord.request_stop()
  coord.join(threads)
    
if __name__ == '__main__':
  main()
