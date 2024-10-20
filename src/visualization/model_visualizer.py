import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from data.position import Position
from .event_visualizer import EventVisualizer
from settings import PHI_RANGE, ETA_RANGE, JET_SIZE, MAP_2D_TICKS, ARROWS_NUMBER

phi_range_size = abs(PHI_RANGE[1] - PHI_RANGE[0])

class ModelVisualizer:
  def __init__(self, model):
    self.model = model
  
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
    random_events = [dataset.get_event(test_loader.dataset.indices[index]) for index in random_indeces]
    random_outputs = [outputs[index] for index in random_indeces]
    random_targets = [targets[index] for index in random_indeces]
    sample_event_index = np.random.randint(len(events))

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    self.arrows_on_eta_phi_plot(random_outputs, random_targets, axs[0], color='blue')
    self.sample_event_plot(events[sample_event_index], targets[sample_event_index], outputs[sample_event_index], axs[1])
    self.distances_histogram(outputs, targets, axs[2])
    self.distances_by_pt_plot(outputs, targets, events, axs[3])
    plt.savefig(output_file)
    plt.show()

  def arrows_on_eta_phi_plot (self, starts, ends, ax, **kwargs):
    def arrow_with_color (eta, phi, deta, dphi, **kwargs):
      distance_normalized = min(1, max(0, 0.5 + 0.5 * np.linalg.norm([deta, dphi]) / 2))
      color = (distance_normalized, 0, 1 - distance_normalized, 0.6)
      ax.arrow(eta, phi, deta, dphi, head_width=0.1, head_length=0.1, fc=color, ec=color, **kwargs)

    for start, end in zip(starts, ends):
      start = Position(start[0], start[1])
      end = Position(end[0], end[1])
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
    def distance (start, end):
      start = Position(start[0], start[1])
      end = Position(end[0], end[1])
      return start.distance(end)

    distances = [distance(start, end) for start, end in zip(starts, ends)]
    percent_of_distances_unser_0_2 = len([distance for distance in distances if distance < 0.2]) / len(distances)
    ax.hist(distances, bins=100)
    ax.set_xlabel('distance')
    ax.set_ylabel(f'count ({percent_of_distances_unser_0_2 * 100:.2f}% under 0.2)')

  def distances_by_pt_plot (self, starts, ends, events, ax):
    def distance (start, end):
      start = Position(start[0], start[1])
      end = Position(end[0], end[1])
      return start.distance(end)

    def pt (event):
      # sum of event.true_four_momentum().pt for all taus in the event
      return sum([momentum.p_t for momentum in event.true_four_momentum()])

    distances = [distance(start, end) for start, end in zip(starts, ends)]
    pts = [pt(event) for event in events]
    ax.scatter(pts, distances)
    ax.set_xlabel('pt')
    ax.set_ylabel('distance')