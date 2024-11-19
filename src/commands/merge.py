# merge multiple h5 files into one, keeping the same structure

import h5py
import numpy as np

from settings import ETA_RANGE

FIELDS = ['event', 'tracks', 'clusters', 'truthTaus']

def create_output_file (output_file, input_file):
  print('Loading input file...')
  with h5py.File(input_file, 'r') as input:
    print('Creating output file...')
    with h5py.File(output_file, 'w') as output:
      invalid_indices = get_invalid_indices(input['truthTaus'][:])
      print(f'Dropping {len(invalid_indices)} invalid events ({len(invalid_indices)/len(input["event"])*100:.2f}%)')

      for field in FIELDS:
        print(f'Creating {field} dataset...')
        data = np.delete(input[field][:], invalid_indices, axis=0)
        output.create_dataset(
          field,
          data=data,
          compression="gzip",
          chunks=(1, input[field].shape[1]) if len(input[field].shape) > 1 else (1,),
          maxshape=(None, input[field].shape[1]) if len(input[field].shape) > 1 else (None,),
        )
        del data
  print('Done')

def append_to_output_file (output_file, input_file):
  print('Loading input file...')
  with h5py.File(input_file, 'r') as input:
    print('Loading output file...')
    with h5py.File(output_file, 'a') as output:
      invalid_indices = get_invalid_indices(input['truthTaus'][:])
      print(f'Dropping {len(invalid_indices)} invalid events ({len(invalid_indices)/len(input["event"])*100:.2f}%)')

      for field in FIELDS:
        print(f'Appending {field} dataset...')
        data = np.delete(input[field][:], invalid_indices, axis=0)
        output[field].resize((output[field].shape[0] + data.shape[0]), axis=0)
        output[field][-data.shape[0]:] = data
        del data
  print('Done')

def get_invalid_indices(truthTaus):
  truthTaus_expanded = np.array(truthTaus.tolist())
  num_truthtaus = np.sum(~np.isnan(truthTaus_expanded[:,:,0:truthTaus.shape[1]]), axis=1)
  not_two_truthtaus = np.unique(np.where(num_truthtaus != 2)[0])
  not_two_barrel_Taus = np.unique(np.where(np.abs(truthTaus_expanded[:, :2, 1]) > ETA_RANGE[1])[0])
  return np.unique(np.concatenate((not_two_truthtaus, not_two_barrel_Taus)))

def merge (input_files, output_file, create_output=True):
  print(f'Merging {len(input_files)} files into {output_file}')

  if create_output:
    print(f'Creating output file from {input_files[0]}')
    create_output_file(output_file, input_files[0])

  files_to_add = input_files[1:] if create_output else input_files

  if len(files_to_add) > 0:
    for input_file in files_to_add:
      print(f'Appending {input_file}')
      append_to_output_file(output_file, input_file)

  print('Merging complete')