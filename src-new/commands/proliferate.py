import numpy as np

from settings import PHI_RANGE
from utils import long_operation
from visualization import DatasetVisualizer

def proliferate (dataset, factor):
  initial_count = len(dataset)

  def run (next):
    new_data = { 'event': [], 'clusters': [], 'tracks': [], 'truthTaus': [] }
    flips = 0
    rotations = []
    get = lambda index: { key: dataset.raw_data[key][index] for key in dataset.raw_data }
    add = lambda event: [new_data[key].append(event[key]) for key in new_data]
    for index in range(len(dataset)):
      original = get(index)
      add(original)
      for i in range(factor - 1):
        # deep copy of the event
        copy = { key: np.copy(original[key]) for key in original }
        # if random > 0.5 is odd, flip eta
        if np.random.rand() > 0.5:
          flips += 1
          copy['clusters']['Clusters.calEta'] = -copy['clusters']['Clusters.calEta']
          copy['tracks']['Tracks.eta'] = -copy['tracks']['Tracks.eta']
          copy['truthTaus']['TruthTaus.eta_vis'] = -copy['truthTaus']['TruthTaus.eta_vis']
        # rotate phi by a random angle
        random_angle = np.random.rand() * (PHI_RANGE[1] - PHI_RANGE[0])
        rotations.append(random_angle)
          
        copy['clusters']['Clusters.calPhi'] = rotate(copy['clusters']['Clusters.calPhi'], random_angle)
        copy['tracks']['Tracks.phi'] = rotate(copy['tracks']['Tracks.phi'], random_angle)
        copy['truthTaus']['TruthTaus.phi_vis'] = rotate(copy['truthTaus']['TruthTaus.phi_vis'], random_angle)

        add(copy)
      next()
    return new_data, flips, rotations

  print('Starting proliferation...')
  dataset.raw_data, flips, rotations = long_operation(run, max=len(dataset), message='Proliferating')
  print()
  print('Done.')
  print(f'Proliferated {initial_count} events by factor of {factor} to {len(dataset)}')
  dataset.save(dataset.source_file.replace('.h5', f'_x{factor}.h5'))

  DatasetVisualizer(dataset).show_proliferation(flips, rotations)

def rotate(angles, by):
  new_angles = (angles + by) % (PHI_RANGE[1] - PHI_RANGE[0]) + PHI_RANGE[0]
  for new_angle in new_angles:
    while (new_angle > PHI_RANGE[1]):
      new_angle -= PHI_RANGE[1] - PHI_RANGE[0]
    while (new_angle < PHI_RANGE[0]):
      new_angle += PHI_RANGE[1] - PHI_RANGE[0]
  return new_angles