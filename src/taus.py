import sys
import time
from data.dataset import EventsDataset
from model.main import MainModel
from utils import datafile_path, modelfolder_path

from commands.show import show
from commands.train import train
from commands.eval import evaluate
from commands.detect import detect
from commands.proliferate import proliferate
from commands.merge import merge
from commands.generate_graphs import generate_graphs
from commands.config import config
from commands.checkpoint import checkpoint

def main (args):
  if isinstance(args, str):
    args = args.split(' ')
  command = args[0]

  if command == 'config':
    config(args[1], args[2])
    return

  from settings import DATA_FILE
  params = { key: value for key, value in [variable.split('=') for variable in args[1:] if variable.find('=') != -1] }
  dataset_file = params.get('ext-src', datafile_path(params.get('src', DATA_FILE)))

  if command == 'proliferate':
    factor = int(params.get('factor', 10))
    proliferate(dataset_file, factor)
    return

  if command == 'merge':
    src = params.get('src', '')
    ext_src = params.get('ext-src', '')
    create_output = params.get('create', 'true') == 'true'
    if src:
      input_files = [datafile_path(file) for file in params.get('src', '').split(',')]
    elif ext_src:
      input_files = [file for file in params.get('ext-src', '').split(',')]
    else:
      print('No input files specified')
      return
    output_file = datafile_path(params.get('output', 'merge_' + str(round(time.time() * 1000))))

    merge(input_files, output_file, create_output)
    return

  loading_type = params.get('loading', 'none')
  cache_type = params.get('cache', 'none')
  normalize_fields = params.get('normalize_fields', 'false') == 'true'
  normalize_energy = params.get('normalize_energy', 'true') == 'true'
  dataset = EventsDataset(dataset_file, loading_type=loading_type, cache_type=cache_type, normalize_fields=normalize_fields, normalize_energy=normalize_energy)
  model = params.get('model', 'small')
  use_post_processing = params.get('post_processing', 'false') == 'true'
  dropout_probability = float(params.get('dropout', 0.15))
  module = MainModel(post_processing=(dataset.post_processing if use_post_processing else False), input_channels=dataset.input_channels, model=model, dropout_probability=dropout_probability)

  if command == 'generate-graphs':
    generate_graphs(dataset, module, params)
    return

  if command == 'checkpoint':
    subcommand = args[1]
    checkpoint(subcommand, dataset, module, params)
    return

  if command == 'show':
    scope = args[1]
    subcommand = args[2]
    show(dataset=dataset, model=module, scope=scope, subcommand=subcommand, params=params)
    return

  if command == 'train':
    folder = modelfolder_path(params.get('folder', 'model_' + str(round(time.time() * 1000))))
    train(dataset, module, folder, params)
    return

  if command == 'eval':
    model_file = modelfolder_path(params.get('weights', 'model_' + str(round(time.time() * 1000))))
    output_folder = params.get('output', 'eval_' + str(round(time.time() * 1000)))
    evaluate(dataset, module, model_file, output_folder, params)
    return

  if command == 'detect':
    model_file = modelfolder_path(params.get('weights', 'model_' + str(round(time.time() * 1000))))
    detect(dataset, module, model_file)
    return

  print(f'Unknown command: {command}')

if __name__ == '__main__':
  main(sys.argv[1:])