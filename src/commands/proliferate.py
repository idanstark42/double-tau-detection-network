import numpy as np

from settings import PHI_RANGE
from utils import long_operation, transform_into_range
from visualization import DatasetVisualizer

def proliferate (dataset, factor):
  initial_count = len(dataset)
  final_count = len(dataset) * factor
  copied_count = final_count - initial_count
  flip_flags = np.random.rand(copied_count) > 0.5
  rotations = np.random.rand(copied_count) * (PHI_RANGE[1] - PHI_RANGE[0])

  def run (next):
    new_data = {
      'event': [None] * len(dataset) * factor,
      'clusters': [None] * len(dataset) * factor,
      'tracks': [None] * len(dataset) * factor,
      'truthTaus': [None] * len(dataset) * factor
    }

    def get (index):
      return { key: dataset.raw_data[key][index] for key in dataset.raw_data }
  
    def generate_copies (original, event_index):
      copies = [{ key: np.copy(original[key]) for key in original } for _ in range(factor)]
      for copy_index, copy in enumerate(copies):
        index = event_index * (factor - 1) + copy_index
        flip(copy, flip_flags[index])
        rotate(copy, rotations[index])
      
      next()
      return copies

    for i in range(len(dataset)):
      original = get(i)
      copies = generate_copies(original, i)

      for key in dataset.raw_data:
        new_data[key][i*factor] = original[key]
        new_data[key][i*factor+1:(i+1)*factor] = [copy[key] for copy in copies]

    return new_data

  print('Starting proliferation...')
  dataset.raw_data = long_operation(run, max=len(dataset), message='Proliferating', concurrent=True)
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