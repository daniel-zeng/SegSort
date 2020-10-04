"""Training script for multi-gpu training PSPNet/ResNet-101 with SegSort."""

from __future__ import print_function

import argparse
import math
import os
import time
import utils.general


import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
import torchvision.transforms as transforms

from seg_models.imagenet_embed_reader import ImageNetEmbedReader
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
  parser.add_argument('--learning_rate', type=float, default=2.5e-4,
                      help='Base learning rate.')
  parser.add_argument('--power', type=float, default=0.9,
                      help='Decay for poly learing rate policy.')
  parser.add_argument('--momentum', type=float, default=0.9,
                      help='Momentum component of the optimiser.')
  parser.add_argument('--weight_decay', type=float, default=5e-4,
                      help='Regularisation parameter for L2-loss.')
  parser.add_argument('--num_classes', type=int, default=1000,
                      help='Number of classes to predict.')
  parser.add_argument('--num_steps', type=int, default=20000,
                      help='Number of training steps.')
  parser.add_argument('--iter_size', type=int, default=10,
                      help='Number of iteration to update weights')
  parser.add_argument('--random_mirror', action='store_true',
                      help='Whether to randomly mirror the inputs.')
  parser.add_argument('--random_crop', action='store_true',
                      help='Whether to randomly crop the inputs.')
  parser.add_argument('--random_scale', action='store_true',
                      help='Whether to randomly scale the inputs.')
  parser.add_argument('--num_loading_workers', type=int, default=10,
                      help='Number of workers to load imagenet.')
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
                      help='Where to save snapshots of the model.')
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

class SimpleNet(nn.Module):
    def __init__(self, h, input_size, output_size, pool="max"):
        super(SimpleNet, self).__init__()
        self.pool = nn.MaxPool2d(h, stride=h) \
          if pool == 'max' else nn.AvgPool2d(h, stride=h)
        self.fc1 = nn.Linear(input_size, output_size)

    def forward(self, x):
      # print(x.size())
      out = x.permute(0, 3, 1, 2) #nhwc to nchw
      # print(out.size())
      out = self.pool(out) 
      out = torch.flatten(out, start_dim = 1)
      # print(out.size())
      # out = torch.max(x, (2, 3)) # Max pooling
      out = self.fc1(out)

      # print("\tIn Model: input size", x.size(), # splits by batch size, sanity check works
      #         "output size", out.size())
      return out

def main():
  print("IMG_NET")

  """Create the model and start the training."""

  # Read CL arguments and snapshot the arguments into text file.
  args = get_arguments()
  utils.general.snapshot_arg(args)
    
  # # The segmentation network is stride 8 by default.
  h, w = map(int, args.input_size.split(','))
  input_size = (h, w)
     
  reader = ImageNetEmbedReader(os.path.join(args.data_dir, "train"), 
    args.batch_size, h, args.num_loading_workers, True)
  print("Total Imgs: {}, Num Batches: {}".format(reader.total_imgs, reader.num_batches))

  # a = reader.dequeue()
  #returns a[0] = torch.Size([256, 60, 60, 32])
  #        a[1] = torch.Size([256])
  # however, norm of last dim is not 1, can be like 5


  # build the rest of the pipeline!
  # simple linear model
  # have the labels
  # compute accuracy

  # now:
  # need to save model periodically

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model = SimpleNet(h, args.embedding_dim, args.num_classes, 'max')
  if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)
  model.to(device) 

  loss_fn = nn.CrossEntropyLoss()
  optimer = optim.Adam(model.parameters(), args.learning_rate)

  pbar = tqdm(range(args.num_steps))
  epoch = 0
  for step in pbar:
    start_time = time.time()
    
    try:
      embeds, labels = reader.dequeue()
    except StopIteration as e:
      reader.reset()
      epoch += 1
      embeds, labels = reader.dequeue()   

    embeds.requires_grad = False
    labels.requires_grad = False
    embeds = embeds.to(device)
    labels = labels.to(device)
    y_pred = model(embeds)
    #should be [batch x 1000]

    loss = loss_fn(y_pred, labels)
    
    optimer.zero_grad()
    loss.backward()
    optimer.step()

    max_index = torch.argmax(y_pred, dim = 1)
    # pdb.set_trace()
    acc = (max_index == labels).double().mean().item()

    duration = time.time() - start_time
    desc = 'loss = {:.3f}, lr = {:.6f}, acc = {:.3f}, epoch = {}'.format(loss, args.learning_rate, acc, epoch)
    pbar.set_description(desc)


  #num supposed images: 1281167
  #num batches: 20019 * 64 = 128121 6(close enough, batch is overest.)
    
if __name__ == '__main__':
  main()
