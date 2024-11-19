from matplotlib import pyplot as plt
import os
from time import time

from .eval import evaluate

from visualization.dataset_visualizer import DatasetVisualizer

from settings import GRAPHS_DIR

DATASET_HISTOGRAMS = [
  'pileup',

  'cluster_count',
  'cluster_cal_e',
  'cluster_pt',
  'cluster_eta_phi',
  'cluster_count_vs_cal_e',
  'cluster_count_vs_pt',
  'cluster_count_vs_pileup',

  'track_count',
  'track_pt',
  'track_eta_phi',
  'track_count_vs_pt',
  'track_count_vs_pileup',

  'truth_count',
  'x_pt',
  'x_eta_phi',
  'taus_delta_r',
  'leading_tau_pt',
  'subleading_tau_pt'
]

def generate_graphs (dataset, module, params):
  # disable plt.show() to avoid blocking the execution. Close the plot instead
  plt.show = lambda: plt.clf()

  events_count = int(params.get('sample-events', 4))

  output_folder = os.path.join(os.path.dirname(__file__), GRAPHS_DIR, params.get('output', str(round(time() * 1000))))
  dataset_folder = os.path.join(output_folder, 'dataset')
  events_folder = os.path.join(dataset_folder, 'events')
  model_folder = os.path.join(output_folder, 'model')

  os.makedirs(events_folder, exist_ok=True)
  os.makedirs(model_folder, exist_ok=True)

  print('1. Generating dataset graphs')
  dataset_visualizer = DatasetVisualizer(dataset)
  dataset_visualizer.sample_random_events(events_count, events_folder)
  dataset_visualizer.multiple_histograms(DATASET_HISTOGRAMS, dataset_folder)

  if params.get('weights', '') == '':
    return
  
  print('2. Generating model graphs')
  evaluate(dataset, module, params.get('weights', ''), model_folder, params)