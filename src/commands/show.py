import matplotlib.pyplot as plt
import numpy as np

import matplotlib.colors as colors
import matplotlib.patches as patches

from data.dataset import *
from utils import *

PLOT_SIZE = 6

def print_fields (dataset):
  print('Cluster fields:')
  [print(python_name) for _, python_name in dataset._cluster_fields]
  print()
  print('Track fields:')
  [print(python_name) for _, python_name in dataset._track_fields]
  print()
  print('Truth fields:')
  [print(python_name) for _, python_name in dataset._truth_fields]

def plot_event_track_histogram (dataset, index, **options):
  event_data = dataset.get_event(index)
  track_positions = np.array([track.position() for track in event_data.tracks if track.eta < 2.5 and track.eta > -2.5 and track.phi < 3.2 and track.phi > -3.2])
  plt.figure(figsize=[PLOT_SIZE, PLOT_SIZE])
  eta_phi_histogram(plt, track_positions, **options)
  plt.show()

def plot_event_cluster_histogram (dataset, index, **options):
  event_data = dataset.get_event(index)
  cluster_positions = np.array([cluster.position() for cluster in event_data.clusters if cluster.cal_eta < 2.5 and cluster.cal_eta > -2.5 and cluster.cal_phi < 3.2 and cluster.cal_phi > -3.2])
  plt.figure(figsize=[PLOT_SIZE, PLOT_SIZE])
  eta_phi_histogram(plt, cluster_positions, **options)
  plt.show()

def plot_event_pt_histogram (dataset, index, **options):
  event_data = dataset.get_event(index)
  (clusters_pts, track_pts) = event_data.clusters_and_tracks_momentum_map(dataset.resolution)
  _fig, axes = plt.subplots(1, 2, figsize=[2 * PLOT_SIZE, PLOT_SIZE])

  for index, topule in enumerate([('Clusters', clusters_pts), ('Tracks', track_pts)]):
    (title, data) = topule
    hist = eta_phi_multivalued_histogram(axes[index], data, **options)
    bounding_boxes = [patches.Circle((eta, phi), 0.2, linewidth=1, edgecolor='red', facecolor='none') for (eta, phi) in event_data.true_position()]
    for box in bounding_boxes:
      axes[index].add_patch(box)
    axes[index].set_title(title)
    plt.colorbar(hist[3])

  plt.show()

def plot_event_histrograms (dataset, index, **options):
  event_data = dataset.get_event(index)
  track_positions = np.array([track.position() for track in event_data.tracks if track.eta < 2.5 and track.eta > -2.5 and track.phi < 3.2 and track.phi > -3.2])
  cluster_positions = np.array([cluster.position() for cluster in event_data.clusters if cluster.cal_eta < 2.5 and cluster.cal_eta > -2.5 and cluster.cal_phi < 3.2 and cluster.cal_phi > -3.2])
  truth_positions = np.array([truth.visible_position() for truth in event_data.truths if truth.eta_vis < 2.5 and truth.eta_vis > -2.5 and truth.phi_vis < 3.2 and truth.phi_vis > -3.2])
  plt.figure(figsize=[3 * PLOT_SIZE, PLOT_SIZE])
  _fig, axes = plt.subplots(1, 3)
  eta_phi_histogram(axes[0], track_positions, **options)
  eta_phi_histogram(axes[1], cluster_positions, **options)
  eta_phi_histogram(axes[2], truth_positions, **options)
  plt.show()

def plot_clusters_and_tracks_per_event (dataset, **options):
  plt.figure(figsize=[PLOT_SIZE, PLOT_SIZE])
  histograms_across_events(dataset, [lambda event: len(event.clusters), lambda event: len(event.tracks)], **options)
  plt.show()

def plot_average_interactions_per_crossing (dataset, **options):
  plt.figure(figsize=[PLOT_SIZE, PLOT_SIZE])
  histograms_across_events(dataset, [lambda event: event.average_interactions_per_crossing], **options)
  plt.show()

def histograms_across_events (dataset, callbacks, **options):
  # build a histogram for every callback function
  hists = [ [] for _ in range(len(callbacks)) ]
  for index in range(len(dataset)):
    if index % 1000 == 0:
      print(f'event {index}')
    event = dataset.get_event(index)
    for i, callback in enumerate(callbacks):
      hists[i].append(callback(event))

  # plot all histograms
  for hist in hists:
    plt.hist(hist, **options)

def histograms_across_tracks (dataset, callbacks, **options):
  # build a histogram for every callback function
  hists = [ [] for _ in range(len(callbacks))]
  for index in range(len(dataset)):
    if index % 1000 == 0:
      print(f'event {index}')
    event = dataset.get_event(index)
    for i, callback in enumerate(callbacks):
      for track in event.tracks:
        hists[i].append(callback(track))

  # plot all histograms
  for hist in hists:
    plt.hist(hist, **options)
  print('done')

def histograms_across_clusters (dataset, callbacks, **options):
  # build a histogram for every callback function
  hists = [ [] for _ in range(len(callbacks))]
  for index in range(len(dataset)):
    if index % 1000 == 0:
      print(f'event {index}')
    event = dataset.get_event(index)
    for i, callback in enumerate(callbacks):
      for cluster in event.clusters:
        hists[i].append(callback(cluster))

  # plot all histograms
  for hist in hists:
    plt.hist(hist, bins=range(0, 1000, 10), alpha=0.5, **options)

def eta_phi_histogram (axes, data, **options):
  eta_res = 100
  phi_res = 100
  hist = np.zeros([eta_res, phi_res])
  for position in data:
    eta_bin = int((position[0] + 2.5) / 5 * eta_res)
    phi_bin = int((position[1] + 3.2) / 6.4 * phi_res)
    hist[eta_bin, phi_bin] += 1

  return eta_phi_multivalued_histogram(axes, hist, **options)

def eta_phi_multivalued_histogram (axes, data, **options):
  etas = []
  phis = []
  weights = []
  for line_index, line in enumerate(data):
    for cell_index, cell in enumerate(line):
      if not cell == 0:
        etas.append(line_index / 100 * 5 - 2.5)
        phis.append(cell_index / 100 * 6.4 - 3.2)
        weights.append(cell)
  return axes.hist2d(etas, phis, weights=weights, bins=100, range=[[-2.5,2.5], [-3.2,3.2]], norm=colors.PowerNorm(0.5), **options)

def number_of_tracks_and_average_interactions_heatmap (dataset, **options):
  tracks_res = 120
  plot_average_interactions_per_crossing_res = 120
  hist = np.zeros([tracks_res, plot_average_interactions_per_crossing_res])
  for index in range(len(dataset)):
    event = dataset.get_event(index)
    if index % 100 == 0:
      print(f'event {index}')
    tracks_bin = int((len(event.tracks) - 1) / 1200 * tracks_res)
    average_interactions_per_crossing_bin = int(event.average_interactions_per_crossing)
    hist[tracks_bin, average_interactions_per_crossing_bin] += 1

  _fig, ax = plt.subplots(figsize=[10,10])

  ax.imshow(hist,
            origin='lower',
            interpolation='bilinear',
            **options
            )

  y_locs, y_labels = plt.yticks()
  y_labels = [int(float(loc / 120) * 1200) for loc in y_locs if loc >= 0 and loc < 120]
  y_locs = [loc for loc in y_locs if loc >= 0 and loc < 120]
  plt.yticks(y_locs, y_labels)
  plt.show()

def clusters_per_pixel (dataset, **options):
  resolution = dataset.resolution
  hist = np.zeros(len(dataset) * resolution * resolution)
  for index in range(len(dataset)):
    event = dataset.get_event(index)
    if index % 100 == 0:
      print(f'event {index}')
    for cluster in event.clusters:
      x, y = relative_position(cluster.position())
      if (x > 0 and x < 1 and y > 0 and y < 1):
        hist[index * resolution * resolution + int(x * resolution) * resolution + int(y * resolution)] += 1
  
  print('show histogram')
  plt.hist(hist, bins=range(0, 10, 1), **options)
  plt.show()

def show (dataset, graph, params):
  if graph == 'fields':
    print_fields(dataset)
    return
  
  if graph == 'event_track_histogram':
    plot_event_track_histogram(dataset, int(params[0]))
    return
  
  if graph == 'event_cluster_histogram':
    plot_event_cluster_histogram(dataset, int(params[0]))
    return
  
  if graph == 'event_histograms':
    plot_event_histrograms(dataset, int(params[0]))
    return
  
  if graph == 'clusters_and_tracks_per_event':
    plot_clusters_and_tracks_per_event(dataset, log=True)
    return
  
  if graph == 'average_interactions_per_crossing':
    plot_average_interactions_per_crossing(dataset, log=True)
    return
  
  if graph == 'number_of_tracks_and_average_interactions_heatmap':
    number_of_tracks_and_average_interactions_heatmap(dataset)
    return
  
  if graph == 'clusters_per_pixel':
    clusters_per_pixel(dataset, log=True)
    return
  
  if graph == 'event_pt_histogram':
    plot_event_pt_histogram(dataset, int(params[0]))
    return