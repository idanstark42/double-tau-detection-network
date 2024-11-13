import torch
import matplotlib.pyplot as plt

from visualization import EventVisualizer, DatasetVisualizer, ModelVisualizer

def with_checkpoint (model, params):
  if 'checkpoint' in params:
    checkpoint = torch.load(params['checkpoint'], map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
  return model

commands = {
  'dataset': {
    '_visualizer': lambda dataset, _model, _params: DatasetVisualizer(dataset),
    'histogram': lambda visualizer, params: visualizer.histogram(visualizer.histogram_fields[params['field']]),
    'fields': lambda visualizer, _params: visualizer.print_fields()
  },
  'event': {
    '_test': lambda dataset, _model, params: 'Event not in dataset' if int(params['event']) > len(dataset) else None,
    '_visualizer': lambda dataset, _model, params: EventVisualizer(dataset.get_event(int(params['event']))),
    'density_map': lambda visualizer, _params: visualizer.density_map(),
    'momentum_map': lambda visualizer, _params: visualizer.momentum_map()
  },
  'checkpoint': {
    '_visualizer': lambda _dataset, model, params: ModelVisualizer(with_checkpoint(model, params)),
    'parameters': lambda visualizer, params: visualizer.parameters_histogram(params['output']),
  }
}

def show (dataset=None, model=None, scope='non', subcommand='non', params={}):
  if scope not in commands:
    exit(f'Unknown scope: {scope}')

  if commands[scope].get('_test'):
    error = commands[scope]['_test'](dataset, model, params)
    if error:
      exit(error)
  
  visualizer = commands[scope]['_visualizer'](dataset, model, params)
  if subcommand not in commands[scope]:
    exit(f'Unknown command: {subcommand} for {scope}')

  commands[scope][subcommand](visualizer, params)

  if params['output']:
    plt.savefig(params['output'])