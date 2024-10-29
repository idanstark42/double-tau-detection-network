# merge multiple h5 files into one, keeping the same structure

import h5py

from utils import long_operation, datafile_path

def create_output_file (output_file, input_file):
  with h5py.File(input_file, 'r') as input:
    event = input['event'][:]
    tracks = input['tracks'][:]
    cells = input['clusters'][:]
    truthTaus = input['truthTaus'][:]
    with h5py.File(output_file, 'w') as output:
      output.create_dataset(
        "event",
        data=event,
        compression="gzip",
        chunks=True,
        maxshape=(None,),
      )
      output.create_dataset(
        "tracks",
        data=tracks,
        compression="gzip",
        chunks=True,
        maxshape=(None, tracks.shape[1]),
      )
      output.create_dataset(
        "clusters",
        data=cells,
        compression="gzip",
        chunks=True,
        maxshape=(None, cells.shape[1]),
      )
      output.create_dataset(
        "truthTaus",
        data=truthTaus,
        compression="gzip",
        chunks=True,
        maxshape=(None, cells.shape[1]),
      )

def append_to_output_file (output_file, input_file):
  with h5py.File(input_file, 'r') as input:
    event = input['event'][:]
    tracks = input['tracks'][:]
    cells = input['clusters'][:]
    truthTaus = input['truthTaus'][:]
    with h5py.File(output_file, 'a') as output:
      output['event'].resize((output['event'].shape[0] + event.shape[0]), axis=0)
      output['event'][-event.shape[0]:] = event
      output['tracks'].resize((output['tracks'].shape[0] + tracks.shape[0]), axis=0)
      output['tracks'][-tracks.shape[0]:] = tracks
      output['clusters'].resize((output['clusters'].shape[0] + cells.shape[0]), axis=0)
      output['clusters'][-cells.shape[0]:] = cells
      output['truthTaus'].resize((output['truthTaus'].shape[0] + truthTaus.shape[0]), axis=0)
      output['truthTaus'][-truthTaus.shape[0]:] = truthTaus

def merge (input_files, output_file):
  print(f'Merging {len(input_files)} into {output_file}')
  
  def merge_h5_files(next):
    create_output_file(output_file, input_files[0])
    next()
    for input_file in input_files[1:]:
      append_to_output_file(output_file, input_file)
      next()

  long_operation(merge_h5_files, max=len(input_files))
  print('Merging complete')