import numpy as np

from settings import PHI_RANGE
from utils import long_operation
from visualization import DatasetVisualizer

def proliferate (dataset, factor):
  initial_count = len(dataset)

  def run (next):
    new_data = {
      'event': [None] * len(dataset) * factor,
      'clusters': [None] * len(dataset) * factor,
      'tracks': [None] * len(dataset) * factor,
      'truthTaus': [None] * len(dataset) * factor
    }
    flips = 0
    rotations = []

    def get (index):
      return { key: dataset.raw_data[key][index] for key in dataset.raw_data }
  
    def generate_copies (original):
      copies = [{ key: np.copy(original[key]) for key in original } for _ in range(factor)]
      for copy in copies:
        # if random > 0.5 is odd, flip eta
        if np.random.rand() > 0.5:
          flips += 1
          flip(copy)

        # rotate phi by a random angle
        random_angle = np.random.rand() * (PHI_RANGE[1] - PHI_RANGE[0])
        rotations.append(random_angle)
        rotate(copy, random_angle)
      
      next()
      return copies

    for i in range(len(dataset)):
      original = get(i)
      copies = generate_copies(original)
      new_data['event'][i*factor] = original['event']
      new_data['clusters'][i*factor] = original['clusters']
      new_data['tracks'][i*factor] = original['tracks']
      new_data['truthTaus'][i*factor] = original['truthTaus']

      new_data['event'][i*factor+1:(i+1)*factor] = [copy['event'] for copy in copies]
      new_data['clusters'][i*factor+1:(i+1)*factor] = [copy['clusters'] for copy in copies]
      new_data['tracks'][i*factor+1:(i+1)*factor] = [copy['tracks'] for copy in copies]
      new_data['truthTaus'][i*factor+1:(i+1)*factor] = [copy['truthTaus'] for copy in copies]
    return new_data, flips, rotations

  print('Starting proliferation...')
  dataset.raw_data, flips, rotations = long_operation(run, max=len(dataset), message='Proliferating')
  print()
  print('Done.')
  print(f'Proliferated {initial_count} events by factor of {factor} to {len(dataset)}')
  dataset.save(dataset.source_file.replace('.h5', f'_x{factor}.h5'))

  DatasetVisualizer(dataset).show_proliferation(flips, rotations)

def flip (event):
  event['Clusters.calEta'] = -event['Clusters.calEta']
  event['Tracks.eta'] = -event['Tracks.eta']
  event['TruthTaus.eta_vis'] = -event['TruthTaus.eta_vis']

def rotate (event, by):
  event['Clusters.calPhi'] = rotate_angles(event['Clusters.calPhi'], by)
  event['Tracks.phi'] = rotate_angles(event['Tracks.phi'], by)
  event['TruthTaus.phi_vis'] = rotate_angles(event['TruthTaus.phi_vis'], by)


def rotate_angles(angles, by):
  return (angles + by) % (PHI_RANGE[1] - PHI_RANGE[0]) + PHI_RANGE[0]