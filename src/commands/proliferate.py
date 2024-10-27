import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager

from settings import PHI_RANGE
from utils import long_operation, transform_into_range
from visualization import DatasetVisualizer

def proliferate (dataset, factor):
  print('Starting proliferation')
  initial_count = len(dataset)
  final_count = len(dataset) * factor
  copied_count = final_count - initial_count
  manager = Manager()

  flip_flags = np.random.rand(copied_count) > 0.5
  rotations = np.random.rand(copied_count) * (PHI_RANGE[1] - PHI_RANGE[0])

  keys = list(dataset.raw_data.keys())
  values = zip(*dataset.raw_data.values())

  shared_keys = manager.list(keys)
  
  def load_events (next):
    with ProcessPoolExecutor() as executor:
      events = [run_with_next(lambda: executor.submit(build_event_dict, shared_keys, manager.list(value)), next) for value in values]
      return [event.result() for event in as_completed(events)]
  
  original_events = long_operation(load_events, max=len(dataset), message='Loading events', multiprocessing=True)
  
  shared_original_events = manager.list(original_events)
  shared_flip_flags = manager.list(flip_flags)
  shared_rotations = manager.list(rotations)

  def generate_copies (next):
    with ProcessPoolExecutor() as executor:
      copies = [run_with_next(lambda: executor.submit(copies_for_event, index, factor, shared_original_events, shared_flip_flags, shared_rotations), next) for index in range(len(dataset))]
      return [copy.result() for copy in as_completed(copies)]

  copied_events = long_operation(generate_copies, max=len(dataset), message='Generating copies')
  shared_copied_events = manager.list(copied_events)
  
  def get_new_data (next):
    new_data = {
      'event': [None] * len(dataset) * factor,
      'clusters': [None] * len(dataset) * factor,
      'tracks': [None] * len(dataset) * factor,
      'truthTaus': [None] * len(dataset) * factor
    }

    for key in new_data:
      for index in range(len(dataset)):
        new_data[key][index * factor] = shared_original_events[index][key]
        for copy_index, copy in enumerate(shared_copied_events[index]):
          new_data[key][index * factor + copy_index + 1] = copy[key]
      next()

    return new_data

  new_data = long_operation(get_new_data, max=len(dataset), message='Setting new data', multiprocessing=True)
  breakpoint()
  for key in keys:
    dataset.raw_data[key] = new_data[key]
  print()
  print('Done.')
  print(f'Proliferated {initial_count} events by a factor of {factor} to {len(dataset)}')
  dataset.save(dataset.source_file.replace('.h5', f'_x{factor}.h5'))

  DatasetVisualizer(dataset).show_proliferation(len([flipping for flipping in flip_flags if flipping]), rotations)

def run_with_next (operation, next):
  future = operation()
  future.add_done_callback(lambda _: next())
  return future

def build_event_dict (keys, value):
  return dict(zip(keys, value))

def copies_for_event (event_index, factor, original_events, flip_flags, rotations):
  event = original_events[event_index]
  copies = [{ key: np.copy(event[key]) for key in event } for _ in range(factor - 1)]
  for copy_index, copy in enumerate(copies):
    index = event_index * (factor - 1) + copy_index
    flip(copy, flip_flags[index])
    rotate(copy, rotations[index])
  return copies

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

# def proliferate (dataset, factor):
#   output_file = dataset.source_file.replace('.h5', f'_x{factor}.h5')
#   initial_count = len(dataset)

#   flips = np.random.rand(initial_count * (factor - 1)) > 0.5
#   rotations = np.random.rand(initial_count * (factor - 1)) * (PHI_RANGE[1] - PHI_RANGE[0])
#   keys = list(dataset.raw_data.keys())

#   with h5py.File(output_file, 'w') as output:
#     print('Initializing output file')
#     for key in dataset.raw_data:
#       output.create_dataset(key, data=dataset.raw_data[key], compression='gzip', chunks=True)
#       output[key].resize((output[key].shape[0] * factor), axis=0)
    
#     print('Running copies')
#     for copy_index in range(factor - 1):
#       def run (next):
#         manager = Manager()
#         shared_data = manager.dict({ key: dataset.raw_data[key] for key in keys })
#         shared_keys = manager.list(keys)
#         with ProcessPoolExecutor() as executor:
#             futures = [run_with_next(lambda: executor.submit(transform, index, shared_data, shared_keys, flips[copy_index * len(dataset) + index], rotations[copy_index * len(dataset) + index]), next) for index in range(len(dataset))]
#             copies = [future.result() for future in as_completed(futures)]
#             for key in keys:
#               dataset.raw_data[key][copy_index * len(dataset):(copy_index + 1) * len(dataset)] = copies[key]

#       long_operation(run, max=len(dataset), message=f'Proliferating {copy_index + 1}', multiprocessing=True)

#   print('Done.')
#   print(f'Proliferated {initial_count} events by a factor of {factor} to {len(dataset)}')
#   dataset.save(dataset.source_file.replace('.h5', f'_x{factor}.h5'))
#   DatasetVisualizer(dataset).show_proliferation(len([flipping for flipping in flips if flipping]), rotations)

# def transform (event_index, original_data, keys, flipping, rotation):
#   copy = { key: np.copy(original_data[key][event_index]) for key in keys }
#   flip(copy, flipping)
#   rotate(copy, rotation)
#   return copy 