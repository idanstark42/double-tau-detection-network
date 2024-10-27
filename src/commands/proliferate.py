import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from settings import PHI_RANGE
from utils import long_operation, transform_into_range
from visualization import DatasetVisualizer

def proliferate (dataset, factor):
  print('Starting proliferation...')
  initial_count = len(dataset)
  final_count = len(dataset) * factor
  copied_count = final_count - initial_count
  print('Generating flags...')
  flip_flags = np.random.rand(copied_count) > 0.5
  rotations = np.random.rand(copied_count) * (PHI_RANGE[1] - PHI_RANGE[0])

  print('inverting')
  keys = dataset.raw_data.keys()
  values = zip(*dataset.raw_data.values())
  def load_events (next):
    def build_event_dict (v):
      event_dict = dict(zip(keys, v))
      next()
      return event_dict
    
    with ThreadPoolExecutor() as executor:
      events = [executor.submit(build_event_dict, v) for v in values]
      return [event for event in as_completed(events)]
  original_events = long_operation(load_events, max=len(dataset), message='Loading events')
  print('inverted')

  def generate_copies (next):
    def copies_for_event (event_index):
      event = original_events[event_index]
      copies = [{ key: np.copy(event[key]) for key in event } for _ in range(factor)]
      for copy_index, copy in enumerate(copies):
        index = event_index * (factor - 1) + copy_index
        flip(copy, flip_flags[index])
        rotate(copy, rotations[index])
      next()
      return copies
    
    with ThreadPoolExecutor() as executor:
      copies = [executor.submit(copies_for_event, i) for i in range(initial_count)]
      return [copy for copy in as_completed(copies)]

  copied_events = long_operation(generate_copies, max=len(dataset), message='Generating copies')
  
  def get_new_data (next):
    new_data = {
      'event': [None] * len(dataset) * factor,
      'clusters': [None] * len(dataset) * factor,
      'tracks': [None] * len(dataset) * factor,
      'truthTaus': [None] * len(dataset) * factor
    }

    def set_events (index):
      original_event = original_events[index]
      for key in dataset.raw_data:
        new_data[key][index*factor] = original_event[key]
        new_data[key][index*factor+1:(index+1)*factor] = [copy[key] for copy in copied_events[index]]
        next()

    with ThreadPoolExecutor() as executor:
      [executor.submit(set_events, i) for i in range(len(dataset))]
      # wait for all events to be set
      executor.shutdown(wait=True)

    return new_data

  dataset.raw_data = long_operation(get_new_data, max=len(dataset), message='Setting new data')
  print()
  print('Done.')
  print(f'Proliferated {initial_count} events by factor of {factor} to {len(dataset)}')
  dataset.save(dataset.source_file.replace('.h5', f'_x{factor}.h5'))

  DatasetVisualizer(dataset).show_proliferation(len([flipping for flipping in flip_flags if flipping]), rotations)

def flip (event, should_flip):
  if not should_flip:
    return
  event['clusters']['Clusters.calEta'] = -event['clusters']['Clusters.calEta']
  event['tracks']['Tracks.eta'] = -event['tracks']['Tracks.eta']
  event['truthTaus']['TruthTaus.eta_vis'] = -event['truthTaus']['TruthTaus.eta_vis']

def rotate (event, by):
  event['clusters']['Clusters.calPhi'] = rotate_angles(event['clusters']['Clusters.calPhi'], by)
  event['tracks']['Tracks.phi'] = rotate_angles(event['tracks']['Tracks.phi'], by)
  event['truthTaus']['TruthTaus.phi_vis'] = rotate_angles(event['truthTaus']['TruthTaus.phi_vis'], by)


def rotate_angles(angles, by):
  return transform_into_range(angles + by, PHI_RANGE)