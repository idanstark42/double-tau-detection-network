import numpy as np
import matplotlib.pyplot as plt
import os
from progress.bar import IncrementalBar
from time import time

from settings import DATA_DIR

def python_names_from_dtype(dtype):
  return [python_name_from_dtype_name(name) for name in dtype.names]

def python_name_from_dtype_name(dtype_name):
  name = dtype_name.split('.')[-1]
  if all([c.islower() or c == '_' for c in name]) or all([c.isupper() or c == '_' for c in name]):
    return name.lower()
  new_name = (name[0].lower() + ''.join([c if c.islower() else (c if next_char.isupper() else f'_{c.lower()}') if prev_char.isupper() else (f'_{c}' if next_char.isupper() else f'_{c.lower()}') for prev_char, c, next_char in zip([''] + list(name[1:]), list(name[1:]), list(name[2:]) + [''])])).split('.')[-1].replace('__', '_')
  new_name = new_name.lower()
  return new_name if new_name[0] != '_' else new_name[1:]

def print_map (map):
  print('#' * map.shape[0] + '##')
  for row in map:
    print('#', end='')
    for cell in row:
      print(' ' if cell < 0.5 else round(float(cell)), end='')
    print('#')
  print('#' * map.shape[0] + '##')

def long_operation (operation, **kwargs):
  bar = IncrementalBar(**kwargs)
  start = time()
  def next (step=1):
    bar.next(step)
    percentage = bar.index / bar.max
    elapsed = time() - start
    if elapsed > 5:
      remaining = (1 - percentage) * elapsed / percentage
      bar.suffix = f'{bar.index}/{bar.max} [{percentage * 100:.1f}%%] {seconds_to_time(remaining)}'
    else:
      bar.suffix = f'{bar.index}/{bar.max} [{percentage * 100:.1f}%%]'
  result = operation(next)
  bar.suffix = f'{bar.max}/{bar.max} [100.0%%] done.'
  bar.finish()
  return result

def transform_into_range (x, range):
  return x if range[0] == range[1] else range[0] + (x - range[0]) % (range[1] - range[0])

def seconds_to_time (seconds):
  hours, minutes, seconds = int(seconds // 3600), int(seconds // 60) % 60, int(seconds) % 60
  return f'{hours:02}:{minutes:02}:{seconds:02}' if hours > 0 else f'{minutes:02}:{seconds:02}'

def datafile_path (name):
  # go to parent directory of this file, then go to the data directory and add h5 suffix
  return os.path.join(os.path.dirname(os.path.dirname(__file__)), DATA_DIR, name + '.h5')

def modelfolder_path (name):
  # go to parent directory of this file, then go to the models directory and add pth suffix
  return os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', name)
