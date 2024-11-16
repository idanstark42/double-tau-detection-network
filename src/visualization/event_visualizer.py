from matplotlib import pyplot as plt
from numpy import pi

from settings import MAP_2D_TICKS, ETA_RANGE, PHI_RANGE, HISTOGRAM_BINS

class EventVisualizer:
  def __init__ (self, event, resolution = 100):
    self.event = event
    self.resolution = resolution

  def density_map (self, show_truth=True, ax=None, output_file=None):
    clusters_points = [cluster.position().relative() for cluster in self.event.clusters]
    tracks_points = [track.position().relative() for track in self.event.tracks]
    truth_points = [truth.visible_position().relative() for truth in self.event.truths]

    self.map([clusters_points, tracks_points], scatter=(truth_points if show_truth else None), ax=ax)

  def momentum_map (self, show_truth=True, ax=None, output_file=None):
    cluster_points = [cluster.position().relative() for cluster in self.event.clusters]
    track_points = [track.position().relative() for track in self.event.tracks]
    truth_points = [truth.visible_position().relative() for truth in self.event.truths]

    clusters_momentum = [cluster.momentum().p_t for cluster in self.event.clusters]
    tracks_momentum = [track.momentum().p_t for track in self.event.tracks]

    self.map([cluster_points, track_points], weights=[clusters_momentum, tracks_momentum], scatter=(truth_points if show_truth else None), ax=ax)

  def tracks_by_pt_histogram (self, ax=None, output_file=None):
    self.histogram([track.pt for track in self.event.tracks], ax=ax, label='Tracks')

  def tracks_by_eta_histogram (self, ax=None, output_file=None):
    self.histogram([track.position().eta for track in self.event.tracks], ax=ax, label='Tracks')
  
  def tracks_by_phi_histogram (self, ax=None, output_file=None):
    self.histogram([track.position().phi for track in self.event.tracks], ax=ax, label='Tracks')

  def clusters_by_pt_histogram (self, ax=None, output_file=None):
    self.histogram([cluster.momentum().p_t for cluster in self.event.clusters], ax=ax, label='Clusters')

  def clusters_by_cal_e_histogram (self, ax=None, output_file=None):
    self.histogram([cluster.cal_e for cluster in self.event.clusters], ax=ax, label='Clusters')
  
  def clusters_by_eta_histogram (self, ax=None, output_file=None):
    self.histogram([cluster.position().eta for cluster in self.event.clusters], ax=ax, label='Clusters')

  def clusters_by_phi_histogram (self, ax=None, output_file=None):
    self.histogram([cluster.position().phi for cluster in self.event.clusters], ax=ax, label='Clusters')

  def histogram (self, values, ax=None, output_file=None, **kwargs):
    if ax:
      ax.hist(values, bins=HISTOGRAM_BINS, **kwargs)
    else:
      plt.hist(values, bins=HISTOGRAM_BINS, **kwargs)
      if output_file:
        plt.savefig(output_file)
      plt.show

  def map (self, maps, weights=None, scatter=None, output_file=None, ax=None):
    independent = ax == None
    if independent:
      fig, ax = plt.subplots()
    for index, map in enumerate(maps):
      if weights == None or weights[index] == None:
        ax.hist2d(*zip(*map), bins=self.resolution, range=[[0, 1], [0, 1]], cmap='Blues')
      else:
        ax.hist2d(*zip(*map), bins=self.resolution, range=[[0, 1], [0, 1]], cmap='Blues', weights=weights[index])
        

    if scatter != None:
      ax.scatter(*zip(*scatter), s=30, c='black', marker='x')
    # set the axis to show the full eta and phi range
    ax.set_xticks([i / MAP_2D_TICKS for i in range(MAP_2D_TICKS + 1)], [round((ETA_RANGE[0] + i * (ETA_RANGE[1] - ETA_RANGE[0]) / MAP_2D_TICKS) * 10) / 10 for i in range(MAP_2D_TICKS + 1)])
    ax.set_yticks([i / MAP_2D_TICKS for i in range(MAP_2D_TICKS + 1)], [round((PHI_RANGE[0] + i * (PHI_RANGE[1] - PHI_RANGE[0]) / MAP_2D_TICKS) * 10) / 10 for i in range(MAP_2D_TICKS + 1)])
    ax.set_ylabel('phi')
    ax.set_xlabel('eta')

    if output_file:
      plt.savefig(output_file)

    if independent:
      plt.show()

  