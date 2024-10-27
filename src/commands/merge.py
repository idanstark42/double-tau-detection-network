# merge multiple h5 files into one, keeping the same structure

import h5py

def merge_h5_files(input_files, output_file):
  with h5py.File(output_file, 'w') as output_h5:
    for input_file in input_files:
      with h5py.File(input_file, 'r') as input_h5:
        for key in input_h5.keys():
          if key in output_h5:
            output_h5[key].resize(output_h5[key].shape[0] + input_h5[key].shape[0], axis=0)
            output_h5[key][-input_h5[key].shape[0]:] = input_h5[key][:]
          else:
            output_h5.create_dataset(key, data=input_h5[key][:])

if __name__ == '__main__':
  input_files = [
    'mx20.h5',
    'mx30.h5',
    'mx40.h5',
    'mx50.h5',
    'mx60.h5'
  ]
  output_file = 'merged.h5'
  print(f'Merging {len(input_files)} files into {output_file}')
  merge_h5_files(input_files, output_file)
  print('Done')