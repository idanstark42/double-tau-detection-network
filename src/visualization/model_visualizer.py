import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import numpy as np

from data.position import Position
from .event_visualizer import EventVisualizer
from settings import PHI_RANGE, ETA_RANGE, JET_SIZE, MAP_2D_TICKS, ARROWS_NUMBER, HISTOGRAM_BINS
from utils import long_operation

phi_range_size = abs(PHI_RANGE[1] - PHI_RANGE[0])

class ModelVisualizer:
  def __init__(self, model):
    self.model = model

  def show_reconstruction_rate_stats (self, outputs, targets, events, output_folder):
    # like these lines only flat:
    output_positions = [Position(output[0], output[1]) for output in outputs] + [Position(output[2], output[3]) for output in outputs]
    target_positions = [Position(target[0], target[1]) for target in targets] + [Position(target[2], target[3]) for target in targets]
    self.distances_histogram(output_positions, target_positions, plt.gca())
    plt.savefig(os.path.join(output_folder, 'distances_histogram.png'))
    plt.show()

    self.plot_reconstruction_rate_by(output_positions, target_positions, events, lambda event: event.total_visible_four_momentum().p_t, 'X pT', os.path.join(output_folder, 'reconstruction_rate_by_pt.png'))
    self.plot_reconstruction_rate_by(output_positions, target_positions, events, lambda event: event.total_visible_four_momentum().eta, 'X η', os.path.join(output_folder, 'reconstruction_rate_by_eta.png'))
    self.plot_reconstruction_rate_by(output_positions, target_positions, events, lambda event: event.total_visible_four_momentum().phi, 'X φ', os.path.join(output_folder, 'reconstruction_rate_by_phi.png'))
    self.plot_reconstruction_rate_by(output_positions, target_positions, events, lambda event: event.total_visible_four_momentum().m, 'X m', os.path.join(output_folder, 'reconstruction_rate_by_m.png'))

    self.plot_reconstruction_rate_by(output_positions, target_positions, events, lambda event: event.average_interactions_per_crossing, 'average interactions per crossing', os.path.join(output_folder, 'reconstruction_rate_by_interactions.png'))
    self.plot_reconstruction_rate_by(output_positions, target_positions, events, lambda event: event.angular_distance_between_taus(), 'clusters count', os.path.join(output_folder, 'reconstruction_rate_by_clusters_count.png'))
  
  def show_losses(self, losses, output_file):
    plt.plot([loss[0] for loss in losses], label='Train Loss')
    plt.plot([loss[1] for loss in losses], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig(output_file)
    plt.show()

  def plot_results (self, outputs, targets, test_loader, dataset, output_file):
    events = [dataset.get_event(test_loader.dataset.indices[index]) for index in range(len(test_loader.dataset))]
    random_indeces = np.random.choice(len(test_loader.dataset), ARROWS_NUMBER, replace=False)
    sample_event_index = np.random.randint(len(events))

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    self.sample_event_plot(events[sample_event_index], targets[sample_event_index], outputs[sample_event_index], axs[1])
    
    # each output and target is a list of four values, two for each tau. Each tau has an eta and a phi
    output_positions = [Position(output[0], output[1]) for output in outputs] + [Position(output[2], output[3]) for output in outputs]
    target_positions = [Position(target[0], target[1]) for target in targets] + [Position(target[2], target[3]) for target in targets]
    self.distances_histogram(output_positions, target_positions, axs[2])
    self.distances_by_pt_plot(output_positions, target_positions, events, axs[3])

    random_position_indices = np.random.choice(len(output_positions), ARROWS_NUMBER, replace=False)
    random_output_positions = [output_positions[index] for index in random_position_indices]
    random_target_positions = [target_positions[index] for index in random_position_indices]
    self.arrows_on_eta_phi_plot(random_output_positions, random_target_positions, axs[0], color='blue')

    plt.savefig(output_file)
    plt.show()

  def arrows_on_eta_phi_plot (self, starts, ends, ax, **kwargs):
    def arrow_with_color (eta, phi, deta, dphi, **kwargs):
      distance_normalized = min(1, max(0, 0.5 + 0.5 * np.linalg.norm([deta, dphi]) / 2))
      color = (distance_normalized, 0, 1 - distance_normalized, 0.6)
      ax.arrow(eta, phi, deta, dphi, head_width=0.1, head_length=0.1, fc=color, ec=color, **kwargs)

    for start, end in zip(starts, ends):
      if abs(start.phi - end.phi) > phi_range_size / 2:
        deta = end.eta - start.eta
        dphi = end.phi - start.phi + (phi_range_size if start.phi > end.phi else -phi_range_size)
        arrow_with_color(start.eta, start.phi, deta, dphi, **kwargs)
        arrow_with_color(end.eta - phi_range_size, end.phi, deta, dphi, **kwargs)
      else:
        arrow_with_color(start.eta, start.phi, end.eta - start.eta, end.phi - start.phi, **kwargs)

    ax.set_xlabel('eta')
    ax.set_ylabel('phi')
    ax.set_xlim(ETA_RANGE[0], ETA_RANGE[1])
    ax.set_ylim(PHI_RANGE[0], PHI_RANGE[1])
    ax.set_xticks([round((ETA_RANGE[0] + i * (ETA_RANGE[1] - ETA_RANGE[0]) / MAP_2D_TICKS) * 10) / 10 for i in range(MAP_2D_TICKS + 1)], [round((ETA_RANGE[0] + i * (ETA_RANGE[1] - ETA_RANGE[0]) / MAP_2D_TICKS) * 10) / 10 for i in range(MAP_2D_TICKS + 1)])
    ax.set_yticks([round((PHI_RANGE[0] + i * (PHI_RANGE[1] - PHI_RANGE[0]) / MAP_2D_TICKS) * 10) / 10 for i in range(MAP_2D_TICKS + 1)], [round((PHI_RANGE[0] + i * (PHI_RANGE[1] - PHI_RANGE[0]) / MAP_2D_TICKS) * 10) / 10 for i in range(MAP_2D_TICKS + 1)])

  def sample_event_plot (self, event, target, output, ax):
    EventVisualizer(event).density_map(show_truth=False, ax=ax)
    circle_width = JET_SIZE / (ETA_RANGE[1] - ETA_RANGE[0])
    circle_height = JET_SIZE / (PHI_RANGE[1] - PHI_RANGE[0])
    for i in range(0, len(target), 2):
      ax.add_patch(patches.Ellipse(Position(target[i], target[i+1]).relative(), circle_width, circle_height, color='red', fill=False))
    for i in range(0, len(output), 2):
      ax.add_patch(patches.Ellipse(Position(output[i], output[i+1]).relative(), circle_width, circle_height, color='blue', fill=False))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

  def distances_histogram (self, starts, ends, ax):
    distances = [start.distance(end) for start, end in zip(starts, ends)]
    percent_of_distances_unser_0_2 = len([distance for distance in distances if distance < 0.2]) / len(distances)
    ax.hist(distances, bins=100)
    ax.set_xlabel('distance')
    ax.set_ylabel(f'count ({percent_of_distances_unser_0_2 * 100:.2f}% under 0.2)')

  def distances_by_pt_plot (self, starts, ends, events, ax):
    distances = [start.distance(end) for start, end in zip(starts, ends)]
    pts = [event.total_visible_four_momentum().p_t for event in events] * 2
    ax.scatter(pts, distances, s=1)
    ax.set_xlabel('pt')
    ax.set_ylabel('distance')
  
  def parameters_histogram (self, output_file):
    def get_params(next):
      parameters = []
      for parameter in self.model.conv_layers.parameters():
        if parameter.requires_grad:
          parameters.append(parameter.data.cpu().numpy().flatten())
          next()
      for parameter in self.model.linear_layers.parameters():
        if parameter.requires_grad:
          parameters.append(parameter.data.cpu().numpy().flatten())
          next()
      return parameters
    
    convulational_params, linear_params = self.model.parameter_counts()
    parametrers = long_operation(get_params, max=convulational_params + linear_params, message='Loading parameters')
    
    plt.hist(parametrers, bins=50)
    plt.yscale('log')
    plt.savefig(output_file)
    plt.show()

  def plot_reconstruction_rate_by (self, outputs, targets, events, get, label, output_file):
    field_values = [get(event) for event in events]
    hist = [0] * HISTOGRAM_BINS
    bin_sizes = [0] * HISTOGRAM_BINS
    for output, target, field_value in zip(outputs, targets, field_values):
      reconstruction_success = Position(output[0], output[1]).distance(Position(target[0], target[1])) < 0.2 and Position(output[2], output[3]).distance(Position(target[2], target[3])) < 0.2
      bin_index = int((field_value - min(field_values)) / (max(field_values) - min(field_values)) * HISTOGRAM_BINS)
      hist[bin_index] += 1 if reconstruction_success else 0
      bin_sizes[bin_index] += 1
    
    hist = [100 * hist[i] / bin_sizes[i] if bin_sizes[i] != 0 else 0 for i in range(HISTOGRAM_BINS)]

    plt.bar(hist, bins=HISTOGRAM_BINS)
    plt.xlabel(label)
    plt.ylabel('reconstruction rate (%)')
    plt.ylim(0, 100)
    if output_file:
      plt.savefig(output_file)
    plt.show()