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
    # print(type(pic))

    #tw: all you need to do: subtract mean

    pic = pic.astype(np.float32)
    # pdb.set_trace()
    # print(pic[0, 0])
    #tw: IMG_MEAN = np.array((122.675, 116.669, 104.008), dtype=np.float32)
    pic /= 255 #tw: original net not like this
    pic -= self.mean # already use mean value
    # print(pic[0, 0])
    pic /= self.std #tw: dont do this
    # print(pic[0, 0])
    return np.array(pic)


def simple_collate(batch):
  imgs = np.array([b[0] for b in batch])
  labels = np.array([b[1] for b in batch])
  # for b in batch:
  #   #b is 3 item typle
  #   print(b[0].shape)
  #   print(b[1:3])
  return (imgs, labels)

class ImageNetReader(object):
  """
  Reads images and corresponding into a Tensorflow queue.
  """
  def __init__(self, data_dir, batch_size, input_size, num_workers):

    #todo: replace with imagenet reader
    self.train_dataset = ImageFolderInstance(
      data_dir,
      transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.2,1.)),
        transforms.RandomGrayscale(p=0.2),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomHorizontalFlip(),
        ToNumpy(),
        Normalize(mean=np.array([122.675, 116.669, 104.008])),
    ]))

    # i = 0
    # while True:
    #   print(self.train_dataset[i][1:3])
    #   i += 300
      # print(self.train_dataset[3][1:3])
      # print(self.train_dataset[15][1:3])

    self.train_loader = tudata.DataLoader(
      self.train_dataset, 
      batch_size=batch_size, shuffle=True,
      collate_fn=simple_collate,
      num_workers=num_workers, pin_memory=False
    )

    self.iter = iter(self.train_loader)
    
  def dequeue(self):
    return next(self.iter)