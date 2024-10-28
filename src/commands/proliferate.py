import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
import h5py
import time

from settings import PHI_RANGE
from utils import long_operation, transform_into_range, seconds_to_time
from visualization import DatasetVisualizer

def proliferate (dataset, factor):
  print('Proliferating.')
  start = time.time()
  output_file = dataset.source_file.replace('.h5', f'_x{factor}.h5')
  initial_count = len(dataset)

  flips = np.random.rand(initial_count * (factor - 1)) > 0.5
  rotations = np.random.rand(initial_count * (factor - 1)) * (PHI_RANGE[1] - PHI_RANGE[0])
  keys = list(dataset.raw_data.keys())

  with h5py.File(output_file, 'w') as output:
    print('Initializing output file')
    for key in dataset.raw_data:
      print(f'Creating dataset for {key}')
      dataset_creation_start_time = time.time()
      output.create_dataset(key, data=dataset.raw_data[key], compression='gzip', chunks=True, maxshape=(None, *dataset.raw_data[key].shape[1:]))
      output[key].resize((output[key].shape[0] * factor), axis=0)
      print(f'Created dataset for {key} in {seconds_to_time(time.time() - dataset_creation_start_time)}')
    output_file_time = time.time()
    print(f'Initialized output file in {seconds_to_time(output_file_time - start)}')
    
    print('Generating copies')
    copy_start_time = time.time()
    def run (next):
      manager = Manager()
      shared_data = manager.dict({ key: list(dataset.raw_data[key]) for key in keys })
      shared_keys = manager.list(keys)
      def sharable_list_from_indices (list, indices):
        return manager.list([[list[index] + len(dataset) * copy_index for index in indices] for copy_index in range(factor - 1)].flatten())
      with ProcessPoolExecutor() as executor:
          # split the dataset into chunks and process them in parallel
          chunk_size = 1000
          chunks = [range(index, min(index + chunk_size, len(dataset))) for index in range(0, len(dataset), chunk_size)]
          futures = [run_with_next(lambda: executor.submit(transform_multiple, indices, factor, shared_data, shared_keys, sharable_list_from_indices(flips, indices), sharable_list_from_indices(rotations, indices)), next) for indices in chunks]
          copy_chunks = [future.result() for future in as_completed(futures)]
          for key in keys:
            for copy_cunk_index, copy_chunk in enumerate(copy_chunks):
              start_index = len(dataset) + copy_cunk_index * chunk_size * factor
              end_index = start_index + len(copy_chunk[key])
              output[key][start_index:end_index] = copy_chunk[key]

    long_operation(run, max=len(dataset), message=f'Proliferating', multiprocessing=True)
    print(f'Generated copies in {seconds_to_time(time.time() - copy_start_time)}')

  print()
  print(f'Done in {seconds_to_time(time.time() - start)}')
  print(f'Proliferated {initial_count} events by a factor of {factor} to {len(dataset)}')
  DatasetVisualizer(dataset).show_proliferation(len([flipping for flipping in flips if flipping]), rotations)

def run_with_next (operation, next):
  future = operation()
  future.add_done_callback(lambda _: next())
  return future

def transform_multiple (indices, shared_data, shared_keys, flips, rotations):
  events = [transform(index, shared_data, shared_keys, flips[index], rotations[index]) for index in indices]
  return { key: np.concatenate([event[key] for event in events], axis=0) for key in shared_keys }

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
