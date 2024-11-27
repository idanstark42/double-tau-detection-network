import numpy as np
import os
from math import inf as Infinity

from commands.train import train

def tune_hyperparameters (dataset, model, model_folder, options):
  search_space = {
    "learning_rate": np.logspace(-5, -1, num=10),
    "weight_decay": np.logspace(-5, -1, num=10),
    "batch_size": np.array([32, 64, 128, 256, 512])
  }

  print("Tuning hyperparameters.")
  print("Search space:")
  for key, values in search_space.items():
    print(f"  {key}: {values}")

  # Define the training function
  def train_model (config):
    config = { key: value for key, value in zip(search_space.keys(), config) }
    submodel_folder = os.path.join(model_folder, '-'.join([f"{key}({value})" for key, value in config.items()]))
    return train(dataset, model, submodel_folder, { **options, **config, })
  
  print("Creating grid search space.")
  grid = np.array(np.meshgrid(*search_space.values())).T.reshape(-1, len(search_space))

  best_loss = Infinity
  best_config = None
  losses = []
  print("Starting grid search.")
  for config in grid:
    loss = (config, train_model(config))
    losses.append((loss, config))
    if loss < best_loss:
      best_loss = loss
      best_config = config

  # print the best configuration
  print('Done.')
  print()
  print(f"Best configuration (loss: {best_loss}):")
  for key, value in best_config.items():
    print(f"  {key}: {value}")
    
