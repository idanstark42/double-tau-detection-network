import os
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

  print('Initializing output file')
  with h5py.File(output_file, 'w') as output:
    for key in dataset.raw_data:
      if output.get(key) and output[key].shape[0] == len(dataset) * factor:
        print(f'Dataset for {key} already exists in output file')
      else:
        print(f'Creating dataset for {key}')
        dataset_creation_start_time = time.time()
        output.create_dataset(key, data=dataset.raw_data[key][:], compression='gzip', chunks=True, maxshape=(None, *dataset.raw_data[key].shape[1:]))
        output[key].resize((output[key].shape[0] * factor), axis=0)
        print(f'Created dataset for {key} in {seconds_to_time(time.time() - dataset_creation_start_time)}')
    output_file_time = time.time()
    print(f'Initialized output file in {seconds_to_time(output_file_time - start)}')

  chunk_size = 1000
  print('creating chunks')
  chunks = [range(index, min(index + chunk_size, len(dataset))) for index in range(0, len(dataset), chunk_size)]
  print('chunks created')

  manager = Manager()
  file_access_lock = manager.Lock()
  # print('Creating shared data')
  # shared_data = manager.dict({ key: list(dataset.raw_data[key]) for key in keys })
  file = dataset.source_file
  print('Creating shared keys')
  shared_keys = manager.list(keys)
  print('Creating shared flips and rotations')
  sharable_flips = manager.list(flips)
  sharable_rotations = manager.list(rotations)

  def create_copies (next):
    with ProcessPoolExecutor(max_workers=int(os.cpu_count() / 2)) as executor:
      futures = [run_with_next(lambda: executor.submit(transform_chunk, indices, factor, len(dataset), file, file_access_lock, shared_keys, sharable_flips, sharable_rotations), next) for indices in chunks]
      return [future.result() for future in as_completed(futures)]
    
  print('Generating copies')
  copy_chunks = create_copies(lambda _: None)

  def add_copies (next):
    with h5py.File(output_file, 'a') as output:
      for index, chunk in enumerate(copy_chunks):
        for key in keys:
          start_index = len(dataset) + index * chunk_size * (factor - 1)
          output[key][start_index:start_index + chunk_size * (factor - 1)] = chunk[key]

  print('Saving copies')
  add_copies(lambda _: None)

  print()
  print(f'Done in {seconds_to_time(time.time() - start)}')
  print(f'Proliferated {initial_count} events by a factor of {factor} to {len(dataset)}')
  DatasetVisualizer(dataset).show_proliferation(len([flipping for flipping in flips if flipping]), rotations)

def run_with_next (operation, next):
  future = operation()
  future.add_done_callback(lambda _: next())
  return future

def transform_chunk (indices, factor, dataset_length, source_file, sourcce_file_access_lock, keys, flips, rotations):
  print(f'Starting chunk {indices[0]}-{indices[-1]}', flush=True)
  flips = extended_list_from_indices(flips, factor, dataset_length, indices)
  rotations = extended_list_from_indices(rotations, factor, dataset_length, indices)
  result = { key: [None] * len(indices) * (factor - 1) for key in keys }
  for lists_index, data_index in enumerate(indices):
    for copy_index in range(factor - 1):
      copy = transform(data_index, source_file, sourcce_file_access_lock, keys, flips[lists_index * (factor - 1) + copy_index], rotations[lists_index * (factor - 1) + copy_index])
      for key in keys:
        result[key][lists_index * (factor - 1) + copy_index] = copy[key]
  print(f'Finished chunk {indices[0]}-{indices[-1]}', flush=True)
  return result

def extended_list_from_indices (list, factor, dataset_length, indices):
  return [list[index] + dataset_length * copy_index for index in indices for copy_index in range(factor - 1)]

def transform (event_index, source_file, sourcce_file_access_lock, keys, flipping, rotation):
  with sourcce_file_access_lock:
    with h5py.File(source_file, 'r') as original_data:
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
