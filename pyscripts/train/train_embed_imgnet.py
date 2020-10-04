"""Training script for multi-gpu training PSPNet/ResNet-101 with SegSort."""

from __future__ import print_function

import argparse
import math
import os
import time
import utils.general
from utils.logger import Logger


import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
import torchvision.transforms as transforms

from seg_models.imagenet_embed_reader import ImageNetEmbedReader
from tqdm import tqdm

import random
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
  parser.add_argument('--num_epochs', type=int, default=300,
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
  parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
  parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')

  # SegSort parameters.
  parser.add_argument('--embedding_dim', type=int, default=32,
                      help='Dimension of the feature embeddings.')
  
  # Misc paramters.
  parser.add_argument('--restore_from', type=str, default='',
                      help='Where restore checkpoint/model parameters from.')
  parser.add_argument('--save_pred_every', type=int, default=10000,
                      help='Save summaries and checkpoint every often.')
  parser.add_argument('--update_tb_every', type=int, default=20,
                      help='Update summaries every often.')
  parser.add_argument('--snapshot_dir', type=str, default='',
                      help='Where to save snapshots of the model.')
  parser.add_argument('--not_restore_classifier', action='store_true',
                      help='Whether to not restore classifier layers.')

  return parser.parse_args()


def save_checkpoint(state, snapshot_dir, filename='checkpoint.pth.tar'):
    filepath = os.path.join(snapshot_dir, filename)
    torch.save(state, filepath)

def adjust_learning_rate(lr, optimizer, epoch, schedule):
  if epoch in schedule:
      lr *= args.gamma
      for param_group in optimizer.param_groups:
        param_group['lr'] = lr
  return lr

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
  print("IMG_NET EMBED TRAIN")

  """Create the model and start the training."""

  # Read CL arguments and snapshot the arguments into text file.
  args = get_arguments()
  torch.manual_seed(args.random_seed)
  torch.cuda.manual_seed_all(args.random_seed)
  np.random.seed(args.random_seed)
  random.seed(args.random_seed)

  utils.general.snapshot_arg(args)
    
  # # The segmentation network is stride 8 by default.
  h, w = map(int, args.input_size.split(','))
  input_size = (h, w)
    
    
  # Create Data Reader
  train_reader = ImageNetEmbedReader(os.path.join(args.data_dir, "train"), 
    args.batch_size, h, args.num_loading_workers, True)
  print("Train: Total Imgs: {}, Num Batches: {}".format(train_reader.total_imgs, train_reader.num_batches))

  # a = reader.dequeue()
  # returns a[0] = torch.Size([256, 60, 60, 32])
  #         a[1] = torch.Size([256])
  # however, norm of last dim is not 1, can be like 5

  # done:
  # build the rest of the pipeline!
  # simple linear model
  # have the labels
  # compute accuracy
  # need to save model periodically
  # weight decay
  # scheduled loss

  # now:
  # similar to bearpaw - add checkpointing and specific train fxn
  

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model = SimpleNet(h, args.embedding_dim, args.num_classes, 'max')
  if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    # model = nn.parallel.DistributedDataParallel(model) # not sure how to use this
    model = nn.DataParallel(model) 
  model.to(device)
  
  lr = args.learning_rate
  loss_fn = nn.CrossEntropyLoss().cuda()
  optimer = optim.Adam(model.parameters(), 
    lr = args.learning_rate, weight_decay = args.weight_decay)

  start_epoch = 0
  if args.restore_from is not None and len(args.restore_from) > 0:
    # Load checkpoint.
    print('Resuming from checkpoint...')
    assert os.path.isfile(args.restore_from), 'Error: no checkpoint directory found!'
    checkpoint_dir = os.path.dirname(args.restore_from)
    checkpoint = torch.load(args.restore_from)
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimer.load_state_dict(checkpoint['optimizer_dict'])
    logger = Logger(os.path.join(args.checkpoint_dir, 'log.txt'), resume=True)
  else:
    logger = Logger(os.path.join(args.snapshot_dir, 'log.txt'))
    logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])


  best_acc = 0
  for epoch in range(start_epoch, args.num_epochs):
    lr = adjust_learning_rate(lr, optimer, epoch, args.schedule)

    print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.num_epochs, lr))
    tr_loss, tr_acc = train(train_reader, model, loss_fn, optimer, epoch, lr, device)
    best_acc = max(best_acc, tr_acc)

    if epoch * args.save_pred_every == 0 and epoch > 0:
      save_checkpoint({
          'epoch': epoch + 1,
          'state_dict': model.state_dict(),
          'acc': tr_acc,
          'best_acc': best_acc,
          'optimizer_dict' : optimizer.state_dict(),
      }, args.snapshot_dir)

  #num supposed images: 1281167
  #num batches: 20019 * 64 = 128121 6(close enough, batch is overest.)


def train(reader, model, loss_fn, optimer, epoch, lr, device): #one epoch
  loss = acc = 0
  pbar = tqdm(reader.loader)
  for i, data in enumerate(pbar):
    start_time = time.time()

    embeds, labels = data
    
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
    desc = 'loss = {:.3f}, lr = {:.6f}, acc = {:.3f}, epoch = {}'.format(loss, lr, acc, epoch)
    pbar.set_description(desc)

  return loss, acc


if __name__ == '__main__':
  main()
