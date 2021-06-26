import torch
from torch import nn
import numpy as np
import random


def train_val_test(file, ratio=(0.6, 0.2, 0.2)):
  """ Generate list of keys for file based on index values
  Input arguments:
    file : h5py File object
          Loaded using h5py.File(path : string)
    ratio : tuple(int), default = (0.6,0.2,0.2)
            Ratios of train/val/test for splitting data in h5py File.
  """

  keys = [key.split('_') for key in file.keys()]
  num_subjects = int(keys[-1][1])
  assert 1.0 * sum(ratio) == 1.0
  sum_ratio = list(ratio)
  if sum_ratio[1] > 0:
    sum_ratio[1] += sum_ratio[0]
  if sum_ratio[2] > 0:
    sum_ratio[2] += sum_ratio[1]
  scaled_ratio = [int(round(val * num_subjects)) for val in sum_ratio]

  ix = np.linspace(0, num_subjects, num_subjects+1, dtype=int)
  train_ix = ix[:scaled_ratio[0]]
  if scaled_ratio[1] > 0:
    val_ix = ix[scaled_ratio[0]:scaled_ratio[1]]
  else:
    val_ix = 0
  if scaled_ratio[2] > 0:
    test_ix = ix[scaled_ratio[1]:scaled_ratio[2]]
  else:
    test_ix = 0

  return train_ix, val_ix, test_ix


def key_gen(file, ix):
  """ Generate list of keys for file based on index values
  Input arguments:
    file : h5py File object
           Loaded using h5py.File(path : string)
    ix : list[int]
         List of index values to find file keys for.
  """
  keys = list(file.keys())
  split_keys = [key.split('_') for key in keys]
  new_keys = []

  for i, key in enumerate(split_keys):
    if int(key[1]) in ix:
      new_keys.append(keys[i])

  return new_keys


class H5DataLoader(torch.utils.data.Dataset):
  def __init__(self, file, keys=None, label='random'):
    """ Dataloader for hdf5 files.
    Input arguments:
      file : h5py File object
             Loaded using h5py.File(path : string)
      keys : list, default = None
             Keys from h5py file to use. Useful for train-val-test split.
             If None, keys generated from entire file.
      label : string, default = 'random'
             Method for loading segmentation labels.
             Options:
                    * 'random' - randomly select one of the available labels
                    * 'vote' - calculate pixel-wise majority vote from available labels
    """
    
    super().__init__()

    self.file = file
    if not keys:
      keys = list(file.keys())
    split_keys = [key.split('_') for key in keys]
    start_subj = int(split_keys[0][1])
    last_subj = int(split_keys[-1][1])
    self.num_subjects = last_subj - start_subj
    self.subjects = np.linspace(start_subj, last_subj, 
                                self.num_subjects+1, dtype=int)
    self.label = label

  def __len__(self):
    return self.num_subjects
  
  def __getitem__(self, index):
    subj_ix = self.subjects[index]
    label_ix = random.randint(0, 2)
    image = torch.unsqueeze(torch.tensor(
        self.file['frame_%05d' % (subj_ix, 
                                       )][()].astype('float32')), dim=0)

    if self.label == 'random':                                
      label = torch.unsqueeze(torch.tensor(
          self.file['label_%05d_%02d' % (subj_ix, 
                                          label_ix
                                          )][()].astype(int)), dim=0)
    elif self.label == 'vote':
      label_batch = torch.cat([torch.unsqueeze(torch.tensor(
          self.file['label_%05d_%02d' % (subj_ix, label_ix
            )][()].astype('float32')), dim=0) for label_ix in range(3)])
      label_mean = torch.unsqueeze(torch.mean(label_batch, dim=0), dim=0)
      label = torch.round(label_mean).int()
    elif self.label == 'mean':
      label_batch = torch.cat([torch.unsqueeze(torch.tensor(
          self.file['label_%05d_%02d' % (subj_ix, label_ix
            )][()].astype('float32')), dim=0) for label_ix in range(3)])
      label = torch.unsqueeze(torch.mean(label_batch, dim=0), dim=0)
    return(image, label)


class ClassifyDataLoader(torch.utils.data.Dataset):
  def __init__(self, file, keys=None):
    """ Dataloader for hdf5 files, with labels converted to classifier labels
    Input arguments:
      file : h5py File object
             Loaded using h5py.File(path : string)
      keys : list, default = None
             Keys from h5py file to use. Useful for train-val-test split.
             If None, keys generated from entire file.
    """
    
    super().__init__()

    self.file = file
    if not keys:
      keys = list(file.keys())
    split_keys = [key.split('_') for key in keys]
    start_subj = int(split_keys[0][1])
    last_subj = int(split_keys[-1][1])
    self.num_subjects = last_subj - start_subj
    self.subjects = np.linspace(start_subj, last_subj, 
                                self.num_subjects+1, dtype=int)
    num_frames = []
    for subj in range(start_subj, last_subj):
      subj_string = str(subj).zfill(4)
      frames = [key[2] for key in split_keys if key[1] == subj_string]
      num_frames.append(int(frames[-1]))

    self.num_frames = num_frames

  def __len__(self):
        return self.num_subjects
  
  def __getitem__(self, index):
    subj_ix = self.subjects[index]
    frame_ix = random.randint(0, self.num_frames[index])
    image = torch.unsqueeze(torch.tensor(
      self.file['frame_%04d_%03d' % (subj_ix, 
                                      frame_ix
                                      )][()].astype('float32')), dim=0)
    label_batch = torch.cat([torch.unsqueeze(torch.tensor(
        self.file['label_%04d_%03d_%02d' % (subj_ix, frame_ix, label_ix
          )][()].astype('float32')), dim=0) for label_ix in range(3)])
    label_vote = torch.sum(label_batch, dim=(1,2))
    sum_vote = torch.sum(label_vote != 0)
    if sum_vote >= 2:
      label = torch.tensor([1.0])
    else:
      label = torch.tensor([0.0])
    return(image, label)

class ClassifyDataLoader_v2(torch.utils.data.Dataset):

  def __init__(self, file, keys=None):
    """ Dataloader for hdf5 files, with labels converted to classifier labels
    Input arguments:
      file : h5py File object
             Loaded using h5py.File(path : string)
      keys : list, default = None
             Keys from h5py file to use. Useful for train-val-test split.
             If None, keys generated from entire file.
    """
    
    super().__init__()

    self.file = file
    if not keys:
      keys = list(file.keys())
    
    self.split_keys = [key.split('_') for key in keys]
    start_subj = int(self.split_keys[0][1])
    last_subj = int(self.split_keys[-1][1])
    self.num_subjects = (last_subj - start_subj)+ 1  #Add 1 to account for 0 idx python
    self.subjects = [key[1] for key in self.split_keys if key[0] == 'frame']
    #self.subjects = np.linspace(start_subj, last_subj, 
    #                            self.num_subjects+1, dtype=int)

  def __len__(self):
        return self.num_subjects
  
  def __getitem__(self, index):
    
    subj_ix = self.subjects[index]
    image_key = 'frame_' + subj_ix
    image = torch.unsqueeze(torch.tensor(self.file[image_key][()].astype('float32')), dim=0)

    label_batch = torch.cat([torch.unsqueeze(torch.tensor(
        self.file[f'label_{subj_ix}_0{label_ix}' ]
        [()].astype('float32')), dim=0) for label_ix in range(3)])
    
    label_vote = torch.sum(label_batch, dim=(1,2))
    sum_vote = torch.sum(label_vote != 0)
    
    #print(sum_vote)
    if sum_vote >= 2:
      label = torch.tensor([1.0])
    else:
      label = torch.tensor([0.0])

    return(image, label)

class TestPlotLoader(torch.utils.data.Dataset):
  def __init__(self, file, keys=None, label='vote'):
    """ Dataloader for hdf5 files, always fixed to first frame to compare plots
    Input arguments:
      file : h5py File object
             Loaded using h5py.File(path : string)
      keys : list, default = None
             Keys from h5py file to use. Useful for train-val-test split.
             If None, keys generated from entire file.
      label : string, default = 'random'
             Method for loading segmentation labels.
             Options:
                    * 'random' - randomly select one of the available labels
                    * 'vote' - calculate pixel-wise majority vote from available labels
    """
    
    super().__init__()

    self.file = file
    if not keys:
      keys = list(file.keys())
    split_keys = [key.split('_') for key in keys]
    start_subj = int(split_keys[0][1])
    last_subj = int(split_keys[-1][1])
    self.num_subjects = last_subj - start_subj
    self.subjects = np.linspace(start_subj, last_subj, 
                                self.num_subjects+1, dtype=int)
    num_frames = []
    for subj in range(start_subj, last_subj):
      subj_string = str(subj).zfill(4)
      frames = [key[2] for key in split_keys if key[1] == subj_string]
      num_frames.append(int(frames[-1]))

    self.num_frames = num_frames
    self.label = label

  def __len__(self):
        return self.num_subjects
  
  def __getitem__(self, index):
    subj_ix = self.subjects[index]
    frame_ix = 0
    label_ix = random.randint(0, 2)
    image = torch.unsqueeze(torch.tensor(
        self.file['frame_%04d_%03d' % (subj_ix, 
                                       frame_ix
                                       )][()].astype('float32')), dim=0)
    if self.label == 'random':                                
      label = torch.unsqueeze(torch.tensor(
          self.file['label_%04d_%03d_%02d' % (subj_ix, 
                                              frame_ix, 
                                              label_ix
                                              )][()].astype(int)), dim=0)
    elif self.label == 'vote':
      label_batch = torch.cat([torch.unsqueeze(torch.tensor(
          self.file['label_%04d_%03d_%02d' % (subj_ix, frame_ix, label_ix
            )][()].astype('float32')), dim=0) for label_ix in range(3)])
      label_mean = torch.unsqueeze(torch.mean(label_batch, dim=0), dim=0)
      label = torch.round(label_mean).int()
    elif self.label == 'mean':
      label_batch = torch.cat([torch.unsqueeze(torch.tensor(
          self.file['label_%04d_%03d_%02d' % (subj_ix, frame_ix, label_ix
            )][()].astype('float32')), dim=0) for label_ix in range(3)])
      label = torch.unsqueeze(torch.mean(label_batch, dim=0), dim=0)
    return(image, label)

class PreScreenLoader(torch.utils.data.Dataset):
  def __init__(self, model, file, keys=None, label='random', threshold=0.5):
    """ Dataloader for hdf5 files.
    Input arguments:
      model : Torch nn.Module object
      file : h5py File object
             Loaded using h5py.File(path : string)
      keys : list, default = None
             Keys from h5py file to use. Useful for train-val-test split.
             If None, keys generated from entire file.
      label : string, default = 'random'
             Method for loading segmentation labels.
             Options:
                    * 'random' - randomly select one of the available labels
                    * 'vote' - calculate pixel-wise majority vote from available labels
    """
    
    super().__init__()

    self.file = file
    if not keys:
      keys = list(file.keys())
    split_keys = [key.split('_') for key in keys]
    start_subj = int(split_keys[0][1])
    last_subj = int(split_keys[-1][1])
    self.num_subjects = last_subj - start_subj
    self.subjects = np.linspace(start_subj, last_subj, 
                                self.num_subjects+1, dtype=int)
    num_frames = []
    for subj in range(start_subj, last_subj):
      subj_string = str(subj).zfill(4)
      frames = [key[2] for key in split_keys if key[1] == subj_string]
      num_frames.append(int(frames[-1]))

    self.num_frames = num_frames
    self.label = label
    self.model = model
    self.threshold = threshold

  def __len__(self):
        return self.num_subjects
  
  def __getitem__(self, index):
    subj_ix = self.subjects[index]
    label_ix = random.randint(0, 2)
    prostate = 'no'
    while prostate == 'no':
      frame_ix = random.randint(0, self.num_frames[index])
      image = torch.unsqueeze(torch.tensor(
          self.file['frame_%04d_%03d' % (subj_ix, 
                                        frame_ix
                                        )][()].astype('float32')), dim=0)
      image_screen = torch.unsqueeze(image, dim=0)
      device = next(self.model.parameters()).device
      image_screen = image_screen.to(device)
      with torch.no_grad():
        self.model.eval()
        screen_pred = self.model(image_screen)
      if screen_pred.item() > self.threshold:
        prostate = 'yes'

    if self.label == 'random':                                
      label = torch.unsqueeze(torch.tensor(
          self.file['label_%04d_%03d_%02d' % (subj_ix, 
                                              frame_ix, 
                                              label_ix
                                              )][()].astype(int)), dim=0)
    elif self.label == 'vote':
      label_batch = torch.cat([torch.unsqueeze(torch.tensor(
          self.file['label_%04d_%03d_%02d' % (subj_ix, frame_ix, label_ix
            )][()].astype('float32')), dim=0) for label_ix in range(3)])
      label_mean = torch.unsqueeze(torch.mean(label_batch, dim=0), dim=0)
      label = torch.round(label_mean).int()
    elif self.label == 'mean':
      label_batch = torch.cat([torch.unsqueeze(torch.tensor(
          self.file['label_%04d_%03d_%02d' % (subj_ix, frame_ix, label_ix
            )][()].astype('float32')), dim=0) for label_ix in range(3)])
      label = torch.unsqueeze(torch.mean(label_batch, dim=0), dim=0)
    return(image, label)
