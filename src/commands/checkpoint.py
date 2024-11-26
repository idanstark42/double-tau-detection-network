import torch
import os

from utils import modelfolder_path

def checkpoint (subcommand, dataset, model, params):
  checkpoint_path = params.get('checkpoint', None)
  if checkpoint_path is None:
    print('No checkpoint path specified')
    return
  
  checkpoint_data = torch.load(checkpoint_path)
  model.load_state_dict(checkpoint_data['model'])

  if subcommand == 'extract':
    # extract the model and save it to the output file
    output_file = modelfolder_path(params.get('output', 'checkpoint_extracted_model_' + checkpoint_data['name']))
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    torch.save(model.state_dict(), output_file)
    return
  
  print(f'Unknown subcommand: {subcommand}')

