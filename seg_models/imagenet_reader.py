import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as tudata
from dataset.folder import ImageFolderInstance

# import pdb

class ToNumpy(object):
  def __call__(self, pic):
    """
    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to numpy.

    Returns:
        Numpy: Converted image.
      """
    return np.array(pic)

class Normalize(object):
  def __init__(self, mean, std = None):
    self.mean = mean
    self.std = std

  def __call__(self, pic):
    """
    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to numpy.

    Returns:
        Numpy: Converted image.
      """

    # All you need to do: subtract mean
    pic = pic.astype(np.float32)
    pic -= self.mean # already use mean value
    return pic


def simple_collate(batch):
  imgs = np.array([b[0] for b in batch])
  labels = np.array([b[1] for b in batch])
  # for b in batch:
  #   #b is 3 item typle
  #   print(b[0].shape)
  #   print(b[1:3])
  return (imgs, labels)

def create_idx_to_class(train_dataset):
  ret_dataset = {}
  for k,v in train_dataset.class_to_idx.items():
    ret_dataset[v] = k
  train_dataset.idx_to_class = ret_dataset

class ImageNetReader(object):
  """
  Reads images and corresponding into a Tensorflow queue.
  """
  def __init__(self, data_dir, batch_size, input_size, num_workers, shuffle=True):

    #todo: replace with imagenet reader
    self.train_dataset = ImageFolderInstance(
      data_dir,
      transforms.Compose([
        transforms.RandomResizedCrop(input_size, 
          scale=(1.,1.),ratio=(0.9, 1.1)),
        
        #don't data-augment for now
        # transforms.RandomResizedCrop(input_size, scale=(0.2,1.)),
        # transforms.RandomGrayscale(p=0.1),
        # transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomHorizontalFlip(),
        ToNumpy(),
        Normalize(mean=np.array([122.675, 116.669, 104.008])),
    ]))

    create_idx_to_class(self.train_dataset)

    # i = 0
    # while True:
    #   print(self.train_dataset[i][1:3])
    #   i += 300
      # print(self.train_dataset[3][1:3])
      # print(self.train_dataset[15][1:3])

    self.train_loader = tudata.DataLoader(
      self.train_dataset, 
      batch_size=batch_size, shuffle=shuffle,
      collate_fn=simple_collate,
      num_workers=num_workers, pin_memory=False
    )

    self.num_batches = len(self.train_loader)
    self.total_imgs = len(self.train_loader.sampler)
    
    self.iter = iter(self.train_loader)

    # i = 0
    # while True:
    #   img, lab = self.dequeue()
    #   if i % 10 == 0:
    #     print(lab)

    #   i += 1
    # print(i)
    # 1/0
    
  def dequeue(self):
    return next(self.iter)

  def get_idx_to_class(self):
    return self.train_dataset.idx_to_class