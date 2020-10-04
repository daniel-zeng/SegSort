import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as tudata
import torchvision.datasets as datasets

# import pdb
NP_EXTS = ('.npy')

def numpy_loader(path):
  with open(path, 'rb') as f:
    return np.load(path)

class NumpyFolder(datasets.DatasetFolder):
  def __init__(self, root, transform=None, target_transform=None,
               loader=numpy_loader, is_valid_file=None):
    super(NumpyFolder, self).__init__(root, loader, NP_EXTS if is_valid_file is None else None,
                                      transform=transform,
                                      target_transform=target_transform,
                                      is_valid_file=is_valid_file)
    self.npys = self.samples

# Likely don't need
def simple_collate(batch):
  imgs = np.array([b[0] for b in batch])
  labels = np.array([b[1] for b in batch])
  return (imgs, labels)

class ImageNetEmbedReader(object):
  """
  Reads images and corresponding into Torch tensors.
  """
  def __init__(self, data_dir, batch_size, input_size, num_workers, shuffle=True):

    #todo: replace with imagenet reader
    self.dataset_folder = NumpyFolder(data_dir)


    self.train_loader = tudata.DataLoader(
      self.dataset_folder, 
      batch_size=batch_size, shuffle=shuffle,
      num_workers=num_workers, pin_memory=True
    )

    self.num_batches = len(self.train_loader)
    self.total_imgs = len(self.train_loader.sampler)
    
    self.iter = iter(self.train_loader) # Should return torch Tensor pinned in cuda memory

    print(self.total_imgs, self.num_batches)

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