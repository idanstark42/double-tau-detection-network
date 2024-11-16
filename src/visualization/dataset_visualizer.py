from matplotlib import pyplot as plt
import numpy as np
import os

from .event_visualizer import EventVisualizer
from settings import HISTOGRAM_BINS, RESOLUTION, MAX_HISTOGRAM_SIZE
from utils import long_operation, python_name_from_dtype_name

class DatasetVisualizer:
  def __init__ (self, dataset):
    self.dataset = dataset

  def print_fields (self):
    for field in self.dataset.dataset_fields:
      print(f'{python_name_from_dtype_name(field)} fields:')
      [print(python_name) for _, python_name in self.dataset._fields[f'{field}_fields']]
      print()

  def show_proliferation (self, copy_count, flips, rotations):
    fig, axes = plt.subplots(1, 2)
    non_flips = copy_count - flips
    axes[0].pie([flips, non_flips], labels=[f'{flips} events eta flipped', f'{non_flips} events eta not flipped'], autopct='%1.1f%%')
    axes[0].set_title('Flips')
    axes[1].hist(rotations, bins=HISTOGRAM_BINS, edgecolor='black')
    axes[1].set_title('Phi Rotations')
    plt.show()

  def random_events (self, count, output_folder):
    random_indeces = np.random.choice(len(self.dataset), count, replace=False)
    os.makedirs(output_folder, exist_ok=True)
    for i in random_indeces:
      event = self.dataset.get_event(i)
      visualizer = EventVisualizer(event)
      visualizer.density_map(output_file=os.path.join(output_folder, f'event_{i}_density_map.png'))
      visualizer.tracks_by_pt_histogram(output_file=os.path.join(output_folder, f'event_{i}_tracks_by_pt_histogram.png'))
      visualizer.tracks_by_eta_histogram(output_file=os.path.join(output_folder, f'event_{i}_tracks_by_eta_histogram.png'))
      visualizer.tracks_by_phi_histogram(output_file=os.path.join(output_folder, f'event_{i}_tracks_by_phi_histogram.png'))
      visualizer.clusters_by_pt_histogram(output_file=os.path.join(output_folder, f'event_{i}_clusters_by_pt_histogram.png'))
      visualizer.clusters_by_cal_e_histogram(output_file=os.path.join(output_folder, f'event_{i}_clusters_by_cal_e_histogram.png'))
      visualizer.clusters_by_eta_histogram(output_file=os.path.join(output_folder, f'event_{i}_clusters_by_eta_histogram.png'))
      visualizer.clusters_by_phi_histogram(output_file=os.path.join(output_folder, f'event_{i}_clusters_by_phi_histogram.png'))


  def histogram (self, config, output_file):
    fields = config['fields']
    callback = config['callback']

    if len(self.dataset) > MAX_HISTOGRAM_SIZE:
      print(f'Dataset too large. Using only partial dataset for histogram of {MAX_HISTOGRAM_SIZE} events.')
      random_indeces = np.random.choice(len(self.dataset), MAX_HISTOGRAM_SIZE, replace=False)

    indicies = random_indeces if len(self.dataset) > MAX_HISTOGRAM_SIZE else range(len(self.dataset))
    def load (next):
      hist = { field: [] for field in fields }
      for i in indicies:
        event = self.dataset.get_event(i)
        datum = callback(event)
        for field in fields:
          hist[field] += datum[field]
        next()
      return hist
    
    result = long_operation(load, max=len(indicies), message='Loading data for histogram')
    if len(fields) == 1:
      plt.hist(result[fields[0]], bins=HISTOGRAM_BINS, edgecolor='black')
      plt.title(fields[0])
      if config.get('x-log', False):
        plt.xscale('log')
      if output_file:
        plt.savefig(output_file)
      plt.show()
      return
    
    if config.get('type', 'side-by-side') == 'side-by-side':
      fig, axes = plt.subplots(1, len(fields))
      for index, field in enumerate(fields):
        hist = np.array(result[field]).flatten().tolist()
        ax = axes[index] if len(fields) > 1 else axes
        ax.hist(hist, bins=HISTOGRAM_BINS, edgecolor='black')
        ax.set_title(field)
        if config.get('x-log', False):
          ax.set_xscale('log')
      if output_file:
        plt.savefig(output_file)
      plt.show()
      return
    
    if config.get('type', 'side-by-side') == '2d' and len(fields) == 2:
      plt.hist2d(result[fields[0]], result[fields[1]], bins=HISTOGRAM_BINS, cmap='Blues', density=True)
      plt.colorbar()
      plt.xlabel(fields[0])
      plt.ylabel(fields[1])
      if output_file:
        plt.savefig(output_file)
      plt.show()
      return

    raise Exception('Unknown histogram type')

  histogram_fields = {
    'average_interaction_per_crossing': {
      'callback': lambda event: { 'average interactions per crossing': [event.average_interactions_per_crossing] },
      'fields': ['average interactions per crossing']
    },

    'cluster_count': {
      'callback': lambda event: { 'cluster count': [len(event.clusters)] },
      'fields': ['cluster count']
    },
    'track_count': {
      'callback': lambda event: { 'track count': [len(event.tracks)] },
      'fields': ['track count']
    },
    'truth_count': {
      'callback': lambda event: { 'truth count': [len(event.truths)] },
      'fields': ['truth count']
    },
    
    'cluster_eta_phi': {
      'callback': lambda event: { 'cluster eta': [cluster.position().eta for cluster in event.clusters], 'cluster phi': [cluster.position().phi for cluster in event.clusters] },
      'fields': ['cluster eta', 'cluster phi'],
      'type': '2d'
    },
    
    'tracks_eta_phi': {
      'callback': lambda event: { 'track eta': [track.position().eta for track in event.tracks], 'track phi': [track.position().phi for track in event.tracks] },
      'fields': ['track eta', 'track phi'],
      'type': '2d'
    },

    'cluster_cal_e': {
      'callback': lambda event: { 'cluster cal_E': [cluster.cal_e for cluster in event.clusters] },
      'fields': ['cluster cal_E']
    },
    'cluster_pt': {
      'callback': lambda event: { 'cluster pt': [cluster.momentum().p_t for cluster in event.clusters] },
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
      'fields': ['clusters mean', 'clusters std', 'tracks mean', 'tracks std'],
      'type': 'side-by-side'
    }
  }