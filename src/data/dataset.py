import h5py
from torch.utils.data import Dataset
import torch
import numpy as np
import gc
import math
from time import time

from data.event import Event
from utils import *
from settings import RESOLUTION, DATASET_FIELDS, ETA_RANGE, PHI_RANGE

class EventsDataset (Dataset):
  def __init__(self, source_file, loading_type='none', cache_type='none', normalize_fields=False):
    super().__init__()
    self.dataset_fields = DATASET_FIELDS
    self.cache_type = cache_type
    self.cache = {}
    self.items_cache = {}
    self.preloaded = False
    self.source_file = source_file
    self.loading_type = loading_type
    self.normalize_fields = normalize_fields
    self.preloading = False
    self.load()
    self.cluster_channels_count = 1
    self.track_channels_count = 1
    self.input_channels = self.cluster_channels_count + self.track_channels_count

  def cluster_channels (self, cluster):
    return [cluster.momentum().p_t]

  def track_channels (self, track):
    return [track.pt]
  
  def get_event(self, index):
    if self.cache_type == 'events' and index in self.cache:
      return self.cache[index]
    
    fields = [(self.data if self.preloaded else self.raw_data)[field][index] for field in self.dataset_fields]

    item = Event(*fields, **self._fields, normalize_fields=self.normalize_fields)

    if self.preloading:
      for i, field in enumerate(self.dataset_fields):
        self.data[field][index] = field[i]

    if self.cache_type == 'events':
      self.cache[index] = item
    return item

  def __getitem__(self, index):
    if self.cache_type == 'items' and index in self.items_cache:
      return self.items_cache[index]
    event = self.get_event(index)

    clusters_map = event.clusters_map(RESOLUTION, lambda cluster: self.cluster_channels(cluster), self.cluster_channels_count)
    tracks_map = event.tracks_map(RESOLUTION, lambda track: self.track_channels(track), self.track_channels_count)
    input = np.concatenate([clusters_map, tracks_map], axis=0)
    target = np.array([position.to_list() for position in event.true_position()], dtype=np.float32).flatten()[:4]
    
    if len(target) < 4:
      target = np.concatenate([target, np.zeros(4 - len(target), dtype=np.float32)])
    
    # if target is all zeros, it means that there is no tau in the event, throw error
    if np.all(target == 0):
      raise ValueError('No tau in the event #{}'.format(index))
      
    # turn inputs and target into torch tensors
    input = torch.tensor(input, dtype=torch.float32)
    target = torch.tensor(target, dtype=torch.float32)

    if self.cache_type == 'items':
      self.items_cache[index] = (input, target)

    return input, target

  def __len__(self):
    return self._length
  
  def __iter__(self):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
      iter_start = 0
      iter_end = len(self)
    else:
      per_worker = int(math.ceil(len(self) / float(worker_info.num_workers)))
      worker_id = worker_info.id
      iter_start = worker_id * per_worker
      iter_end = min(iter_start + per_worker, len(self))
    return iter(range(iter_start, iter_end))
  
  def clear_cache (self):
    # release memorys
    self.cache = {}
    self.items_cache = {}
    gc.collect()
  
  def post_processing(self, x):
    x[0::2] = transform_into_range(x[..., 0::2], ETA_RANGE)
    x[1::2] = transform_into_range(x[..., 1::2], PHI_RANGE)
    return x
  
  # io operations

  def save (self, filename):
    with h5py.File(filename, 'w') as f:
      for key in self.raw_data:
        f.create_dataset(key, data=self.raw_data[key])

  def load(self):
    if self.loading_type == 'none':
      self.raw_data = h5py.File(self.source_file, 'r')
      self._fields = { f'{field}_fields': [(name, python_name_from_dtype_name(name)) for name in self.raw_data[field].dtype.names] for field in self.dataset_fields }
      self._length = len(self.raw_data['event'])
    elif self.loading_type == 'full':
      self.full_preload()
  
  def full_preload (self):
    self.preloaded = True
    self.data = {}
    self._fields = {}
    def preload (next):
      for field in self.dataset_fields:
        self.data[field] = self.raw_data[field][:]
        self._fields[f'{field}_fields'] = [(name, python_name_from_dtype_name(name)) for name in self.raw_data[field].dtype.names]
        next()
    self._length = len(self.data['event'])
    long_operation(preload, max=len(self.dataset_fields), message='Preloading')

  def start_partial_preloading (self):
    self.preloading = True
    self.preloaded = False
    self.raw_data = h5py.File(self.source_file, 'r')
    self._fields = { f'{field}_fields': [(name, python_name_from_dtype_name(name)) for name in self.raw_data[field].dtype.names] for field in self.dataset_fields }
    self.data = { field: {} for field in self.dataset_fields }
    self._length = len(self.raw_data['event'])

  def finish_partial_preloading (self):
    self.preloading = False
    self.preloaded = True
    self.raw_data.close()
    self.raw_data = None
