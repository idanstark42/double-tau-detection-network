from matplotlib import pyplot as plt, colors
import numpy as np
import os

from .event_visualizer import EventVisualizer
from settings import HISTOGRAM_BINS, MAX_HISTOGRAM_SIZE
from utils import long_operation, python_name_from_dtype_name, scatter_histogram

class DatasetVisualizer:
  def __init__ (self, dataset, show=True):
    self.dataset = dataset
    self.show = show

  def print_fields (self):
    for field in self.dataset.dataset_fields:
      print(f'{python_name_from_dtype_name(field)} fields:')
      [print(python_name) for _, python_name in self.dataset._fields[f'{field}_fields']]
      print()

  def show_proliferation (self, copy_count, flips, rotations):
    fig, axes = plt.subplots(1, 2)
    non_flips = copy_count - flips
    axes[0].pie([flips, non_flips], labels=[f'{flips} events eta flipped', f'{non_flips} events eta not flipped'], autopct='%1.1f%%')
    axes[1].hist(rotations, bins=HISTOGRAM_BINS, edgecolor='black')
    self.show_if_should()

  def sample_random_events (self, count, output_folder):
    random_indeces = np.random.choice(len(self.dataset), count, replace=False)
    os.makedirs(output_folder, exist_ok=True)
    print(f'Sampling {count} random events to {output_folder}')
    for i in random_indeces:
      event = self.dataset.get_event(i)
      visualizer = EventVisualizer(event, show=self.show)
      visualizer.density_map(output_file=os.path.join(output_folder, f'event_{i}_density_map.png'))
      visualizer.momentum_map(output_file=os.path.join(output_folder, f'event_{i}_momentum_map.png'))
      visualizer.tracks_by_pt_histogram(output_file=os.path.join(output_folder, f'event_{i}_tracks_by_pt_histogram.png'))
      visualizer.tracks_by_eta_histogram(output_file=os.path.join(output_folder, f'event_{i}_tracks_by_eta_histogram.png'))
      visualizer.tracks_by_phi_histogram(output_file=os.path.join(output_folder, f'event_{i}_tracks_by_phi_histogram.png'))
      visualizer.clusters_by_pt_histogram(output_file=os.path.join(output_folder, f'event_{i}_clusters_by_pt_histogram.png'))
      visualizer.clusters_by_cal_e_histogram(output_file=os.path.join(output_folder, f'event_{i}_clusters_by_cal_e_histogram.png'))
      visualizer.clusters_by_eta_histogram(output_file=os.path.join(output_folder, f'event_{i}_clusters_by_eta_histogram.png'))
      visualizer.clusters_by_phi_histogram(output_file=os.path.join(output_folder, f'event_{i}_clusters_by_phi_histogram.png'))

  def multiple_histograms (self, fields, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    histograms_data = { 'fields': [], 'configs': [], 'callbacks': [], 'histograms': [], 'skipped': [] }
    for field in fields:
      histogram_fields, histogram_field_configs, callback = self.load_histogram_data(self.histogram_fields[field])
      histograms_data['fields'].append(histogram_fields)
      histograms_data['configs'].append(histogram_field_configs)
      histograms_data['callbacks'].append(callback)
      histograms_data['histograms'].append({ field: [] for field in histogram_fields })
      histograms_data['skipped'].append(0)

    indicies = self.histogram_indices()
    
    def load (next):
      for i in indicies:
        event = self.dataset.get_event(i)
        
        for j, field in enumerate(fields):
          if self.histogram_fields[field].get('valid', False) and not self.histogram_fields[field]['valid'](event):
            histograms_data['skipped'][j] += 1
            continue
          datum = histograms_data['callbacks'][j](event)
          for field in histograms_data['fields'][j]:
            if histograms_data['configs'][j][histograms_data['fields'][j].index(field)].get('cross', False) == 'follower':
              histograms_data['histograms'][j][field].append(datum[field])
            else:
              histograms_data['histograms'][j][field] += datum[field]
        next()
      return histograms_data
    
    histograms_data = long_operation(load, max=len(indicies), message='Loading data for histograms')

    for i, field in enumerate(fields):
      self.draw_histogram(histograms_data['fields'][i], histograms_data['configs'][i], histograms_data['histograms'][i], self.histogram_fields[field], os.path.join(output_folder, f'{field}.png'))

  def histogram (self, config, output_file):
    fields, field_configs, callback = self.load_histogram_data(config)
    indicies = self.histogram_indices()

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

    self.draw_histogram(fields, field_configs, result, config, output_file)

  def draw_histogram (self, fields, field_configs, result, config, output_file):
    if len(fields) == 1:
      fig, ax = plt.subplots()
      scatter_histogram(result[fields[0]], ax, HISTOGRAM_BINS, range=field_configs[0].get('xlim', None), type=field_configs[0].get('type', 'percentage'))
      plt.xlabel(fields[0])
      plt.ylabel('Density')
      if 'xlim' in field_configs[0]:
        plt.xlim(field_configs[0]['xlim'])
      if config.get('x-log', False):
        plt.xscale('log')
      if config.get('y-log', False):
        plt.yscale('log')
      if output_file:
        plt.savefig(output_file)
      self.show_if_should()
      return

    if config.get('type', 'side-by-side') == 'side-by-side':
      fig, axes = plt.subplots(1, len(fields))
      for index, field in enumerate(fields):
        hist = np.array(result[field]).flatten().tolist()
        ax = axes[index] if len(fields) > 1 else axes
        scatter_histogram(hist, ax, HISTOGRAM_BINS, range=field_configs[index].get('xlim', None), type=field_configs[index].get('type', 'percentage'))
        ax.set_xlabel(field)
        ax.set_ylabel('Density')
        if config.get('x-log', False):
          ax.set_xscale('log')
        if config.get('y-log', False):
          ax.set_yscale('log')
      if output_file:
        plt.savefig(output_file)
      self.show_if_should()
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
      plt.xlabel(fields[0])
      plt.ylabel(fields[1])
      if output_file:
        plt.savefig(output_file)
      self.show_if_should()
      return

    raise Exception('Unknown histogram type')
  
  def load_histogram_data (self, config):
    fields = config['fields']
    field_configs = [config.get('config', {}).get(field, {}) for field in fields]
    callback = config['callback']
    return fields, field_configs, callback
  
  def histogram_indices (self):
    if len(self.dataset) > MAX_HISTOGRAM_SIZE:
      print(f'Dataset too large. Using only partial dataset for histogram of {MAX_HISTOGRAM_SIZE} events.')
      return np.random.choice(len(self.dataset), MAX_HISTOGRAM_SIZE, replace=False)

    return range(len(self.dataset))

  histogram_fields = {
    'pileup': {
      'callback': lambda event: { 'Pileup': [event.average_interactions_per_crossing] },
      'fields': ['Pileup']
    },

    'cluster_count': {
      'callback': lambda event: { 'Cluster': [len(event.clusters)] },
      'fields': ['Cluster']
    },
    'cluster_cal_e': {
      'callback': lambda event: { 'Cluster Energy [GeV]': [cluster.cal_e / 1000 for cluster in event.clusters] },
      'fields': ['Cluster Energy [GeV]'],
      'config': { 'Cluster Energy [GeV]': { 'y-log': True } }
    },
    'cluster_pt': {
      'callback': lambda event: { 'Cluster pT [GeV]': [cluster.momentum().p_t / 1000 for cluster in event.clusters] },
      'fields': ['Cluster pT [GeV]'],
      'config': { 'Cluster pT [GeV]': { 'y-log': True } }
    },
    'cluster_eta_phi': {
      'callback': lambda event: { 'Cluster η': [cluster.position().eta for cluster in event.clusters], 'Cluster φ': [cluster.position().phi for cluster in event.clusters] },
      'fields': ['Cluster η', 'Cluster φ'],
      'type': '2d'
    },
    'cluster_count_vs_cal_e': {
      'callback': lambda event: { 'Clusters': [len(event.clusters)], 'Cluster Energy [GeV]': [cluster.cal_e / 1000 for cluster in event.clusters] },
      'fields': ['Clusters', 'Cluster Energy [GeV]'],
      'type': '2d',
      'config': { 'Cluster Energy [GeV]': { 'cross': 'leader', 'xlim': [0, 50] }, 'Clusters': { 'cross': 'follower' } }
    },
    'cluster_count_vs_pt': {
      'callback': lambda event: { 'Clusters': [len(event.clusters)], 'Cluster pT [GeV]': [cluster.momentum().p_t / 1000 for cluster in event.clusters] },
      'fields': ['Clusters', 'Cluster pT [GeV]'],
      'type': '2d',
      'config': { 'Cluster pT [GeV]': { 'cross': 'leader', 'xlim': [0, 50] }, 'Clusters': { 'cross': 'follower' } }
    },
    'cluster_count_vs_pileup': {
      'callback': lambda event: { 'Clusters': [len(event.clusters)], 'Pileup': [event.average_interactions_per_crossing] },
      'fields': ['Clusters', 'Pileup'],
      'type': '2d',
      'config': { 'Pileup': { 'cross': 'follower' }, 'Clusters': { 'cross': 'leader' } }
    },

    'track_count': {
      'callback': lambda event: { 'Tracks': [len(event.tracks)] },
      'fields': ['Tracks']
    },
    'track_pt': {
      'callback': lambda event: { 'Track pT [GeV]': [track.pt / 1000 for track in event.tracks] },
      'fields': ['Track pT [GeV]'],
      'config': { 'Track pT [GeV]': { 'y-log': True } }
    },
    'track_eta_phi': {
      'callback': lambda event: { 'Track η': [track.position().eta for track in event.tracks], 'Track φ': [track.position().phi for track in event.tracks] },
      'fields': ['Track η', 'Track φ'],
      'type': '2d'
    },
    'track_count_vs_pt': {
      'callback': lambda event: { 'Tracks': [len(event.tracks)], 'Track pT [GeV]': [track.pt / 1000 for track in event.tracks] },
      'fields': ['Tracks', 'Track pT [GeV]'],
      'type': '2d',
      'config': { 'Track pT [GeV]': { 'cross': 'leader', 'xlim': [0, 50] }, 'Tracks': { 'cross': 'follower' } }
    },
    'track_count_vs_pileup': {
      'callback': lambda event: { 'Tracks': [len(event.tracks)], 'Pileup': [event.average_interactions_per_crossing] },
      'fields': ['Tracks', 'Pileup'],
      'type': '2d',
      'config': { 'Pileup': { 'cross': 'follower' }, 'Tracks': { 'cross': 'leader' } }
    },

    'truth_count': {
      'callback': lambda event: { 'Truth τ Count': [len(event.truths)] },
      'fields': ['Truth τ Count']
    },
    'x_m': {
      'callback': lambda event: { 'X Mass [GeV]': [event.total_visible_four_momentum().m / 1000] },
      'fields': ['X Mass [GeV]']
    },
    'x_pt': {
      'callback': lambda event: { 'X pT [GeV]': [event.total_visible_four_momentum().p_t / 1000] },
      'fields': ['X pT [GeV]']
    },
    'x_eta_phi': {
      'callback': lambda event: { 'X η': [event.total_visible_four_momentum().eta], 'X φ': [event.total_visible_four_momentum().phi] },
      'fields': ['X η', 'X φ'],
      'type': '2d'
    },
    'taus_delta_r': {
      'callback': lambda event: { 'ΔR': [event.angular_distance_between_taus()] },
      'fields': ['ΔR'],
      'valid': lambda event: len(event.truths) == 2
    },
    'leading_tau_pt': {
      'callback': lambda event: { 'Leading τ pT [GeV]': [event.leading_pt() / 1000] },
      'fields': ['Leading τ pT [GeV]']
    },
    'subleading_tau_pt': {
      'callback': lambda event: { 'Subleading τ pT [GeV]': [event.subleading_pt() / 1000] },
      'fields': ['Subleading τ pT [GeV]']
    },

    'normlization_factors': {
      'callback': lambda event: event.normalization_factors(),
      'fields': ['clusters mean', 'clusters std', 'tracks mean', 'tracks std'],
      'type': 'side-by-side'
    }
  }
  
  def show_if_should (self):
    if self.show:
      plt.show()
    else:
      plt.clf()