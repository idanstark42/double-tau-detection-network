import sys
import time
from data.dataset import EventsDataset
from model.main import MainModel
from utils import datafile_path, modelfolder_path

from commands.show import show
from commands.train import train_module
from commands.eval import evaluate
from commands.detect import detect
from commands.proliferate import proliferate
from commands.config import config

if __name__ == '__main__':
  command = sys.argv[1]

  if command == 'config':
    config(sys.argv[2], sys.argv[3])
    exit()

  from settings import DATA_FILE
  params = { key: value for key, value in [variable.split('=') for variable in sys.argv[2:]] }
  dataset_file = datafile_path(params.get('src', DATA_FILE))

  if command == 'proliferate':
    factor = int(params.get('factor', 10))
    proliferate(dataset_file, factor)
    exit()
  
  dataset = EventsDataset(dataset_file)

  if command == 'show':
    scope = sys.argv[2]
    params = sys.argv[3:]
    show(dataset, scope, params)
    exit()

  model = params.get('model', 'small')
  use_post_processing = params.get('post_processing', 'false') == 'true'
  dropout_probability = float(params.get('dropout', 0.15))
  module = MainModel(post_processing=(dataset.post_processing if use_post_processing else False), input_channels=dataset.input_channels, model=model, dropout_probability=dropout_probability)

  if command == 'train':
    output = modelfolder_path(params.get('output', 'model_' + str(round(time.time() * 1000))))
    train_module(dataset, module, output, params)
    exit()

  if command == 'eval':
    model_file = modelfolder_path(params.get('weights', 'model_' + str(round(time.time() * 1000)))) + '\\model.pth'
    evaluate(dataset, module, model_file, params)
    exit()

  if command == 'detect':
    model_file = modelfolder_path(params.get('weights', 'model_' + str(round(time.time() * 1000)))) + '\\model.pth'
    detect(dataset, module, model_file)
    exit()

  print(f'Unknown command: {command}')