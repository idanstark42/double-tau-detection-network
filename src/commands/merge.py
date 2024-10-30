# merge multiple h5 files into one, keeping the same structure

import h5py
import numpy as np
from utils import long_operation

from settings import ETA_RANGE

def create_output_file (output_file, input_file):
  with h5py.File(input_file, 'r') as input:
    event = input['event'][:]
    tracks = input['tracks'][:]
    clusters = input['clusters'][:]
    truthTaus = input['truthTaus'][:]

    event, tracks, clusters, truthTaus = clean(event, tracks, clusters, truthTaus)
  
    with h5py.File(output_file, 'w') as output:
      output.create_dataset(
        "event",
        data=event,
        compression="gzip",
        chunks=(1,),
        maxshape=(None,),
      )
      output.create_dataset(
        "tracks",
        data=tracks,
        compression="gzip",
        chunks=(1, tracks.shape[1]),
        maxshape=(None, tracks.shape[1]),
      )
      output.create_dataset(
        "clusters",
        data=clusters,
        compression="gzip",
        chunks=(1, clusters.shape[1]),
        maxshape=(None, clusters.shape[1]),
      )
      output.create_dataset(
        "truthTaus",
        data=truthTaus,
        compression="gzip",
        chunks=(1, truthTaus.shape[1]),
        maxshape=(None, truthTaus.shape[1]),
      )

def append_to_output_file (output_file, input_file):
  with h5py.File(input_file, 'r') as input:
    event = input['event'][:]
    tracks = input['tracks'][:]
    clusters = input['clusters'][:]
    truthTaus = input['truthTaus'][:]

    event, tracks, clusters, truthTaus = clean(event, tracks, clusters, truthTaus)

    with h5py.File(output_file, 'a') as output:
      output['event'].resize((output['event'].shape[0] + event.shape[0]), axis=0)
      output['event'][-event.shape[0]:] = event
      output['tracks'].resize((output['tracks'].shape[0] + tracks.shape[0]), axis=0)
      output['tracks'][-tracks.shape[0]:] = tracks
      output['clusters'].resize((output['clusters'].shape[0] + clusters.shape[0]), axis=0)
      output['clusters'][-clusters.shape[0]:] = clusters
      output['truthTaus'].resize((output['truthTaus'].shape[0] + truthTaus.shape[0]), axis=0)
      output['truthTaus'][-truthTaus.shape[0]:] = truthTaus

def clean(event, tracks, clusters, truthTaus):
  #Find invalid events (based on ==2 truthTaus with |eta| < 2.5)
  truthTaus_expanded = np.array(truthTaus.tolist())
  num_truthtaus = np.sum(~np.isnan(truthTaus_expanded[:,:,0:truthTaus.shape[1]]), axis=1)
  not_two_truthtaus = np.unique(np.where(num_truthtaus != 2)[0])
  not_two_barrel_Taus = np.unique(np.where(np.abs(truthTaus_expanded[:, :2, 1]) > ETA_RANGE[1])[0])
  invalid_indices = np.unique(np.concatenate((not_two_truthtaus, not_two_barrel_Taus)))
  #Drop invalid events
  event = np.delete(event, invalid_indices, axis=0)
  tracks = np.delete(tracks, invalid_indices, axis=0)
  clusters = np.delete(clusters, invalid_indices, axis=0)
  truthTaus = np.delete(truthTaus, invalid_indices, axis=0)
  return event, tracks, clusters, truthTaus

def merge (input_files, output_file):
  print(f'Merging {len(input_files)} into {output_file}')

  def merge_h5_files(next):
    create_output_file(output_file, input_files[0])
    next()
    if len(input_files) == 1:
      return
    for input_file in input_files[1:]:
      append_to_output_file(output_file, input_file)
      next()

  long_operation(merge_h5_files, max=len(input_files))
  print('Merging complete')