from matplotlib import pyplot as plt, colors
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

  def sample_random_events (self, count, output_folder):
    random_indeces = np.random.choice(len(self.dataset), count, replace=False)
    os.makedirs(output_folder, exist_ok=True)
    for i in random_indeces:
      event = self.dataset.get_event(i)
      visualizer = EventVisualizer(event)
      visualizer.density_map(output_file=os.path.join(output_folder, f'event_{i}_density_map.png'))
      visualizer.momentum_map(output_file=os.path.join(output_folder, f'event_{i}_momentum_map.png'))
      visualizer.tracks_by_pt_histogram(output_file=os.path.join(output_folder, f'event_{i}_tracks_by_pt_histogram.png'))
      visualizer.tracks_by_eta_histogram(output_file=os.path.join(output_folder, f'event_{i}_tracks_by_eta_histogram.png'))
      visualizer.tracks_by_phi_histogram(output_file=os.path.join(output_folder, f'event_{i}_tracks_by_phi_histogram.png'))
      visualizer.clusters_by_pt_histogram(output_file=os.path.join(output_folder, f'event_{i}_clusters_by_pt_histogram.png'))
      visualizer.clusters_by_cal_e_histogram(output_file=os.path.join(output_folder, f'event_{i}_clusters_by_cal_e_histogram.png'))
      visualizer.clusters_by_eta_histogram(output_file=os.path.join(output_folder, f'event_{i}_clusters_by_eta_histogram.png'))
      visualizer.clusters_by_phi_histogram(output_file=os.path.join(output_folder, f'event_{i}_clusters_by_phi_histogram.png'))

  def histogram (self, config, output_file):
    fields = config['fields']
    field_configs = [config.get('config', {}).get(field, {}) for field in fields]
    callback = config['callback']

    if len(self.dataset) > MAX_HISTOGRAM_SIZE:
      print(f'Dataset too large. Using only partial dataset for histogram of {MAX_HISTOGRAM_SIZE} events.')
      random_indeces = np.random.choice(len(self.dataset), MAX_HISTOGRAM_SIZE, replace=False)

    indicies = random_indeces if len(self.dataset) > MAX_HISTOGRAM_SIZE else range(len(self.dataset))
    def load (next):
      skipped = 0
      hist = { field: [] for field in fields }
      for i in indicies:
        event = self.dataset.get_event(i)
        if config.get('valid', False) and not config['valid'](event):
          skipped += 1
          continue
        datum = callback(event)
        for field in fields:
          if field_configs[fields.index(field)].get('cross', False) == 'follower':
            hist[field].append(datum[field])
          else:
            hist[field] += datum[field]
        next()
      return hist, skipped

    result, skipped_count = long_operation(load, max=len(indicies), message='Loading data for histogram')
    print(f'Loaded {len(indicies) - skipped_count} events. Got {len(result[fields[0]])} events for histogram')
    if skipped_count:
      print(f'Skipped {skipped_count} events')

    if len(fields) == 1:
      plt.hist(result[fields[0]], bins=HISTOGRAM_BINS, edgecolor='black', density=True, range=field_configs[0].get('xlim', None))
      plt.title(f'events by {fields[0]}')
      plt.xlabel(fields[0])
      plt.ylabel('events density')
      if 'xlim' in field_configs[0]:
        plt.xlim(field_configs[0]['xlim'])
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
        ax.hist(hist, bins=HISTOGRAM_BINS, edgecolor='black', density=True, range=field_configs[index].get('xlim', None))
        ax.set_title(f'events by {field}')
        ax.set_xlabel(field)
        ax.set_ylabel('events density')
        if config.get('x-log', False):
          ax.set_xscale('log')
      if output_file:
        plt.savefig(output_file)
      plt.show()
      return

    if config.get('type', 'side-by-side') == '2d' and len(fields) == 2:
      hist_x, hist_y = result[fields[0]], result[fields[1]]
      if field_configs[0].get('cross', False) == 'leader':
        hist_x = [x for x, ys in zip(hist_x, hist_y) for _ in range(len(ys))]
        hist_y = [item for sublist in hist_y for item in sublist]
      elif field_configs[1].get('cross', False) == 'leader':
        hist_y = [y for y, xs in zip(hist_y, hist_x) for _ in range(len(xs))]
        hist_x = [item for sublist in hist_x for item in sublist]

      plt.hist2d(hist_x, hist_y, bins=HISTOGRAM_BINS, cmap='Blues', density=True, range=[field_configs[0].get('xlim', None), field_configs[1].get('xlim', None)], norm=colors.LogNorm())
      plt.colorbar()
      plt.title(f'events by {fields[0]} and {fields[1]}')
      plt.xlabel(fields[0])
      plt.ylabel(fields[1])
      if output_file:
        plt.savefig(output_file)
      plt.show()
      return

    raise Exception('Unknown histogram type')

  histogram_fields = {
    'pileup': {
      'callback': lambda event: { 'pileup': [event.average_interactions_per_crossing] },
      'fields': ['pileup']
    },

    'cluster_count': {
      'callback': lambda event: { 'cluster count': [len(event.clusters)] },
      'fields': ['cluster count']
    },
    'cluster_cal_e': {
      'callback': lambda event: { 'cluster cal_E': [cluster.cal_e for cluster in event.clusters] },
      'fields': ['cluster cal_E'],
      'config': { 'cluster cal_E': { 'xlim': [0, 0.5] } }
    },
    'cluster_pt': {
      'callback': lambda event: { 'cluster pt': [cluster.momentum().p_t for cluster in event.clusters] },
      'fields': ['cluster pt'],
      'config': { 'cluster pt': { 'xlim': [0, 0.4] } }
    },
    'cluster_eta_phi': {
      'callback': lambda event: { 'cluster η': [cluster.position().eta for cluster in event.clusters], 'cluster φ': [cluster.position().phi for cluster in event.clusters] },
      'fields': ['cluster η', 'cluster φ'],
      'type': '2d'
    },
    'cluster_count_vs_cal_e': {
      'callback': lambda event: { 'amount': [len(event.clusters)], 'cal_E': [cluster.cal_e for cluster in event.clusters] },
      'fields': ['amount', 'cal_E'],
      'type': '2d',
      'config': { 'cal_E': { 'xlim': [0, 0.5], 'cross': 'follower' }, 'amount': { 'cross': 'leader' } }
    },
    'cluster_count_vs_pt': {
      'callback': lambda event: { 'amount': [len(event.clusters)], 'pT': [cluster.momentum().p_t for cluster in event.clusters] },
      'fields': ['amount', 'pT'],
      'type': '2d',
      'config': { 'pT': { 'xlim': [0, 0.4], 'cross': 'follower' }, 'amount': { 'cross': 'leader' } }
    },
    'cluster_count_vs_pileup': {
      'callback': lambda event: { 'amount': [len(event.clusters)], 'pileup': [event.average_interactions_per_crossing] },
      'fields': ['amount', 'pileup'],
      'type': '2d',
      'config': { 'pileup': { 'cross': 'follower' }, 'amount': { 'cross': 'leader' } }
    },

    'track_count': {
      'callback': lambda event: { 'track count': [len(event.tracks)] },
      'fields': ['track count']
    },
    'track_pt': {
      'callback': lambda event: { 'track pt': [track.pt for track in event.tracks] },
      'fields': ['track pt'],
      'config': { 'track pt': { 'xlim': [0, 10000] } }
    },
    'track_eta_phi': {
      'callback': lambda event: { 'track η': [track.position().eta for track in event.tracks], 'track φ': [track.position().phi for track in event.tracks] },
      'fields': ['track η', 'track φ'],
      'type': '2d'
    },
    'track_count_vs_pt': {
      'callback': lambda event: { 'amount': [len(event.tracks)], 'pT': [track.pt for track in event.tracks] },
      'fields': ['amount', 'pT'],
      'type': '2d',
      'config': { 'pT': { 'xlim': [0, 0.4], 'cross': 'follower' }, 'amount': { 'cross': 'leader' } }
    },
    'track_count_vs_pileup': {
      'callback': lambda event: { 'amount': [len(event.tracks)], 'pileup': [event.average_interactions_per_crossing] },
      'fields': ['amount', 'pileup'],
      'type': '2d',
      'config': { 'pileup': { 'cross': 'follower' }, 'amount': { 'cross': 'leader' } }
    },

    'truth_count': {
      'callback': lambda event: { 'truth count': [len(event.truths)] },
      'fields': ['truth count']
    },
    'truth_x_pt': {
      'callback': lambda event: { 'X pT': [event.total_visible_four_momentum().p_t] },
      'fields': ['X pT']
    },
    'truth_x_eta_phi': {
      'callback': lambda event: { 'X η': [event.total_visible_four_momentum().eta], 'X φ': [event.total_visible_four_momentum().phi] },
      'fields': ['X η', 'X φ'],
      'type': '2d'
    },
    'truth_delta_r': {
      'callback': lambda event: { 'ΔR': [event.angular_distance_between_taus()] },
      'fields': ['ΔR'],
      'valid': lambda event: len(event.truths) == 2
    },
    'truth_leading_pt': {
      'callback': lambda event: { 'leading pT': [event.leading_pt()] },
      'fields': ['leading pT']
    },
    'truth_subleading_pt': {
      'callback': lambda event: { 'subleading pT': [event.subleading_pt()] },
      'fields': ['subleading pT']
    },

    'normlization_factors': {
      'callback': lambda event: event.normalization_factors(),
      'fields': ['clusters mean', 'clusters std', 'tracks mean', 'tracks std'],
      'type': 'side-by-side'
    }
  }