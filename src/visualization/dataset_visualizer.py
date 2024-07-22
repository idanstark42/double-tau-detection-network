from matplotlib import pyplot as plt
import numpy as np

from settings import HISTOGRAM_BINS
from utils import long_operation, python_name_from_dtype_name

class DatasetVisualizer:
  def __init__ (self, dataset):
    self.dataset = dataset

  def print_fields (self):
    for field in self.dataset.dataset_fields:
      print(f'{python_name_from_dtype_name(field)} fields:')
      [print(python_name) for _, python_name in self.dataset._fields[f'{field}_fields']]
      print()

  def histogram (self, config):
    fields = config['fields']
    callback = config['callback']

    def load (next):
      hist = { field: [] for field in fields }
      for i in range(len(self.dataset)):
        event = self.dataset.get_event(i)
        datum = callback(event)
        for field in fields:
          hist[field].append(datum[field])
        next()
      return hist
    
    result = long_operation(load, max=len(self.dataset))
    fig, axes = plt.subplots(1, len(fields))
    for index, field in enumerate(fields):
      hist = np.array(result[field]).flatten().tolist()
      ax = axes[index] if len(fields) > 1 else axes
      ax.hist(hist, bins=HISTOGRAM_BINS, edgecolor='black')
      ax.set_title(field)
    plt.show()

  histogram_fields = {
    'average_interaction_per_crossing': {
      'callback': lambda event: { 'average interaction per crossing': [event.average_interactions_per_crossing] },
      'fields': ['average interactions per crossing']
    },

    'cluster_count': {
      'callback': lambda event: { 'cluster count': len(event.clusters) },
      'fields': ['cluster count']
    },
    'track_count': {
      'callback': lambda event: { 'track count': len(event.tracks) },
      'fields': ['track count']
    },
    'truth_count': {
      'callback': lambda event: { 'truth count': len(event.truths) },
      'fields': ['truth count']
    },

    'cluster_pt': {
      'callback': lambda event: { 'cluster pt': [cluster.pt for cluster in event.clusters] },
      'fields': ['cluster pt']
    },
    'track_pt': {
      'callback': lambda event: { 'track pt': [track.pt for track in event.tracks] },
      'fields': ['track pt']
    },
    'truth_pt': {
      'callback': lambda event: { 'truth pt': [truth.pt for truth in event.truths] },
      'fields': ['truth pt']
    },

    'normlization_factors': {
      'callback': lambda event: event.normalization_factors(),
      'fields': ['clusters mean', 'clusters std', 'tracks mean', 'tracks std']
    }
  }