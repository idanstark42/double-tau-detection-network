from visualization import EventVisualizer, DatasetVisualizer

commands = {
  'dataset': {
    '_visualizer': lambda dataset, _params: DatasetVisualizer(dataset),
    'histogram': lambda visualizer, params: visualizer.histogram(visualizer.histogram_fields[params[0]]),
    'fields': lambda visualizer, _params: visualizer.print_fields()
  },
  'event': {
    '_test': lambda dataset, params: 'Event not in dataset' if int(params[1]) > len(dataset) else None,
    '_visualizer': lambda dataset, params: EventVisualizer(dataset.get_event(int(params[1]))),
    'density_map': lambda visualizer, _params: visualizer.density_map(),
    'momentum_map': lambda visualizer, _params: visualizer.momentum_map()
  }
}

def show (dataset, scope, params):
  if scope not in commands:
    exit(f'Unknown scope: {scope}')

  if commands[scope].get('_test'):
    error = commands[scope]['_test'](dataset, params)
    if error:
      exit(error)
  
  visualizer = commands[scope]['_visualizer'](dataset, params)
  command = params[0]
  if command not in commands[scope]:
    exit(f'Unknown command: {command}')

  commands[scope][command](visualizer, params[1:])