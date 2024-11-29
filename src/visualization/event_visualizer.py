from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from numpy import pi
import matplotlib.patches as patches

from settings import MAP_2D_TICKS, ETA_RANGE, PHI_RANGE, HISTOGRAM_BINS, JET_SIZE
from utils import transparent_cmap

from data.position import Position

class EventVisualizer:
  def __init__ (self, event, resolution = 100, show=True):
    self.event = event
    self.resolution = resolution
    self.show = show

  def density_map (self, show_truth=True, ax=None, output_file=None):
    clusters_points = [cluster.position().relative() for cluster in self.event.clusters]
    tracks_points = [track.position().relative() for track in self.event.tracks]
    truth_points = [truth.visible_position().relative() for truth in self.event.truths]

    self.map([clusters_points, tracks_points], scatter=(truth_points if show_truth else None), ax=ax, title='Clusters and Tracks density by η and φ', output_file=output_file, configs=[{'label': 'Clusters', 'cmap': 'Blues', 'alpha': 0.5}, {'label': 'Tracks', 'cmap': 'Oranges', 'alpha': 0.5}])

  def momentum_map (self, show_truth=True, ax=None, output_file=None):
    cluster_points = [cluster.position().relative() for cluster in self.event.clusters]
    track_points = [track.position().relative() for track in self.event.tracks]
    truth_points = [truth.visible_position().relative() for truth in self.event.truths]

    clusters_momentum = [cluster.momentum().p_t for cluster in self.event.clusters]
    tracks_momentum = [track.momentum().p_t for track in self.event.tracks]

    self.map([cluster_points, track_points], weights=[clusters_momentum, tracks_momentum], scatter=(truth_points if show_truth else None), ax=ax, title='Clusters and Tracks pT density by η and φ', output_file=output_file, configs=[{'label': 'Clusters Momentum', 'cmap': 'Blues', 'alpha': 0.5, 'norm': LogNorm()}, {'label': 'Tracks Momentum', 'cmap': 'Oranges', 'alpha': 0.5, 'norm': LogNorm()}])

  def tracks_by_pt_histogram (self, ax=None, output_file=None, title_addition=''):
    self.histogram([track.pt / 1000 for track in self.event.tracks], ax=ax, title=f'Tracks by pT {title_addition}', x_label='pT [GeV]', yl_label='Density', output_file=output_file)

  def tracks_by_eta_histogram (self, ax=None, output_file=None, title_addition=''):
    self.histogram([track.position().eta for track in self.event.tracks], ax=ax, title=f'Tracks by η {title_addition}', x_label='η', yl_label='Density', output_file=output_file)

  def tracks_by_phi_histogram (self, ax=None, output_file=None, title_addition=''):
    self.histogram([track.position().phi for track in self.event.tracks], ax=ax, title=f'Tracks by φ {title_addition}', x_label='φ', yl_label='Density', output_file=output_file)

  def clusters_by_pt_histogram (self, ax=None, output_file=None, title_addition=''):
    self.histogram([cluster.momentum().p_t / 1000 for cluster in self.event.clusters], ax=ax, title=f'Clusters by pT {title_addition}', x_label='pT [GeV]', yl_label='Density', output_file=output_file)

  def clusters_by_cal_e_histogram (self, ax=None, output_file=None, title_addition=''):
    self.histogram([cluster.cal_e / 1000 for cluster in self.event.clusters], ax=ax, title=f'Clusters by Calorimeter Energy {title_addition}', x_label='Calorimeter Energy [GeV]', yl_label='Density', output_file=output_file)

  def clusters_by_eta_histogram (self, ax=None, output_file=None, title_addition=''):
    self.histogram([cluster.position().eta for cluster in self.event.clusters], ax=ax, title=f'Clusters by η {title_addition}', x_label='η', yl_label='Density', output_file=output_file)

  def clusters_by_phi_histogram (self, ax=None, output_file=None, title_addition=''):
    self.histogram([cluster.position().phi for cluster in self.event.clusters], ax=ax, title=f'Clusters by φ {title_addition}', x_label='φ', yl_label='Density', output_file=output_file)

  def histogram (self, values, title, x_label, yl_label, ax=None, output_file=None, **kwargs):
    if ax:
      ax.hist(values, bins=HISTOGRAM_BINS, edgecolor='black', histtype='step', density=True, **kwargs)
      ax.set_xlabel(x_label)
      ax.set_ylabel(yl_label)
    else:
      plt.hist(values, bins=HISTOGRAM_BINS, edgecolor='black', histtype='step', density=True, **kwargs)
      plt.xlabel(x_label)
      plt.ylabel(yl_label)
      if output_file:
        plt.savefig(output_file)
      self.show_if_should()

  def map (self, maps, weights=None, scatter=None, output_file=None, ax=None, title=None, configs=None):
    independent = ax == None
    if independent:
      fig, ax = plt.subplots()
    for index, map in enumerate(maps):
      config = configs[index] if configs else {}
      label = config.get('label', None)
      if label:
        del config['label']
      
      if weights == None or weights[index] == None:
        ax.hist2d(*zip(*map), bins=self.resolution, range=[[0, 1], [0, 1]], **config)
      else:
        ax.hist2d(*zip(*map), bins=self.resolution, range=[[0, 1], [0, 1]], weights=weights[index], **config)
      
      # colorbar
      colorbar = plt.colorbar(ax.collections[-1], ax=ax)
      if label:
        colorbar.set_label(label)

    if output_file and independent:
      plt.savefig(output_file)


    if scatter != None:
      circle_width = JET_SIZE / (ETA_RANGE[1] - ETA_RANGE[0])
      circle_height = JET_SIZE / (PHI_RANGE[1] - PHI_RANGE[0])
      for point in scatter:
        ax.add_patch(patches.Ellipse(Position(point[0], point[1]).relative(), circle_width, circle_height, color='red', fill=False))
    # set the axis to show the full eta and phi range
    ax.set_xticks([i / MAP_2D_TICKS for i in range(MAP_2D_TICKS + 1)], [round((ETA_RANGE[0] + i * (ETA_RANGE[1] - ETA_RANGE[0]) / MAP_2D_TICKS) * 10) / 10 for i in range(MAP_2D_TICKS + 1)])
    ax.set_yticks([i / MAP_2D_TICKS for i in range(MAP_2D_TICKS + 1)], [round((PHI_RANGE[0] + i * (PHI_RANGE[1] - PHI_RANGE[0]) / MAP_2D_TICKS) * 10) / 10 for i in range(MAP_2D_TICKS + 1)])
    ax.set_ylabel('φ')
    ax.set_xlabel('η')

    if output_file:
      plt.savefig(output_file)

    if independent:
      self.show_if_should()

  def show_if_should (self):
    if self.show:
      plt.show()
    else:
      plt.clf()
