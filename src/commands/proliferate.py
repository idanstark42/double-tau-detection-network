import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
import h5py

from settings import PHI_RANGE
from utils import long_operation, transform_into_range
from visualization import DatasetVisualizer

def proliferate (dataset, factor):
  print('Proliferating.')
  output_file = dataset.source_file.replace('.h5', f'_x{factor}.h5')
  initial_count = len(dataset)

  flips = np.random.rand(initial_count * (factor - 1)) > 0.5
  rotations = np.random.rand(initial_count * (factor - 1)) * (PHI_RANGE[1] - PHI_RANGE[0])
  keys = list(dataset.raw_data.keys())

  with h5py.File(output_file, 'w') as output:
    print('Initializing output file')
    for key in dataset.raw_data:
      print(f'Creating dataset for {key}')
      output.create_dataset(key, data=dataset.raw_data[key], compression='gzip', chunks=True, maxshape=(None, *dataset.raw_data[key].shape[1:]))
      output[key].resize((output[key].shape[0] * factor), axis=0)
    
    print('Generating copies')
    for copy_index in range(factor - 1):
      def run (next):
        manager = Manager()
        shared_data = manager.dict({ key: list(dataset.raw_data[key]) for key in keys })
        shared_keys = manager.list(keys)
        with ProcessPoolExecutor() as executor:
            futures = [run_with_next(lambda: executor.submit(transform, index, shared_data, shared_keys, flips[copy_index * len(dataset) + index], rotations[copy_index * len(dataset) + index]), next) for index in range(len(dataset))]
            copies = [future.result() for future in as_completed(futures)]
            for key in keys:
              dataset.raw_data[key][copy_index * len(dataset):(copy_index + 1) * len(dataset)] = copies[key]

      long_operation(run, max=len(dataset), message=f'Proliferating ({copy_index + 1}/{factor - 1})', multiprocessing=True)

  print('Done.')
  print(f'Proliferated {initial_count} events by a factor of {factor} to {len(dataset)}')
  dataset.save(dataset.source_file.replace('.h5', f'_x{factor}.h5'))
  DatasetVisualizer(dataset).show_proliferation(len([flipping for flipping in flips if flipping]), rotations)

def run_with_next (operation, next):
  future = operation()
  future.add_done_callback(lambda _: next())
  return future

def transform (event_index, original_data, keys, flipping, rotation):
  copy = { key: np.copy(original_data[key][event_index]) for key in keys }
  flip(copy, flipping)
  rotate(copy, rotation)
  return copy

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
