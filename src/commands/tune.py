from ray import tune
import os

from commands.train import train

def tune_hyperparameters (dataset, model, model_folder, options):
  # Define the search space
  search_space = {
    "learning_rate": tune.loguniform(1e-4, 1e-1),
    "weight_decay": tune.loguniform(1e-6, 1e-3),
    "batch_size": tune.choice([32, 64, 128, 256, 512]),
  }

  # Define the training function
  def train_model (config):
    submodel_folder = os.path.join(model_folder, '-'.join([f"{key}({value})" for key, value in config.items()]))
    return train(dataset, model, submodel_folder, { **options, **config, })

  # Start the hyperparameter search
  analysis = tune.run(train_model, config=search_space)

  # Return the best hyperparameters
  return analysis.get_best_config(metric="loss", mode="min")