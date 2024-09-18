import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from data.position import Position
from .event_visualizer import EventVisualizer
from settings import PHI_RANGE, ETA_RANGE, JET_SIZE, MAP_2D_TICKS

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

  def plot_results (self, outputs, targets, events, output_file):
    # draw two plots side by side
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    self.arrows_on_eta_phi_plot(outputs, targets, axs[0], color='blue')
    sample_event_index = np.random.randint(len(events))
    self.sample_event_plot(events[sample_event_index], targets[sample_event_index], outputs[sample_event_index], axs[1])
    self.distances_histogram(outputs, targets, axs[2])
    self.distances_by_pt_plot(outputs, targets, events, axs[3])
    plt.savefig(output_file)
    plt.show()

  def arrows_on_eta_phi_plot (self, starts, ends, ax, **kwargs):
    def arrow_with_color (x, y, dx, dy, **kwargs):
      distance_normalized = min(1, max(0, 0.5 + 0.5 * np.linalg.norm([dx, dy]) / 2))
      color = (distance_normalized, 0, 1 - distance_normalized, 0.6)
      ax.arrow(x, y, dx, dy, head_width=0.1, head_length=0.1, fc=color, ec=color, **kwargs)

    for start, end in zip(starts, ends):
      if abs(start[1] - end[1]) > abs(PHI_RANGE[1] - PHI_RANGE[0]) / 2:
        arrow_with_color(end[1], end[0], end[1]-start[1], end[0] - start[0], **kwargs)
        if start[1] < end[1]:
          arrow_with_color(start[1], start[0], end[1]-start[1] - 2 * np.pi, end[0] - start[0], **kwargs)
        else:
          arrow_with_color(start[1], start[0], end[1]-start[1], end[0] - start[0] - 2 * np.pi, **kwargs)
      else:
        arrow_with_color(start[1], start[0], end[1]-start[1], end[0] - start[0], **kwargs)

    ax.set_xlabel('phi')
    ax.set_ylabel('eta')
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-2.5, 2.5)
    ax.set_xticks([round((ETA_RANGE[0] + i * (ETA_RANGE[1] - ETA_RANGE[0]) / MAP_2D_TICKS) * 10) / 10 for i in range(MAP_2D_TICKS + 1)], [round((ETA_RANGE[0] + i * (ETA_RANGE[1] - ETA_RANGE[0]) / MAP_2D_TICKS) * 10) / 10 for i in range(MAP_2D_TICKS + 1)])
    ax.set_yticks([round((PHI_RANGE[0] + i * (PHI_RANGE[1] - PHI_RANGE[0]) / MAP_2D_TICKS) * 10) / 10 for i in range(MAP_2D_TICKS + 1)], [round((PHI_RANGE[0] + i * (PHI_RANGE[1] - PHI_RANGE[0]) / MAP_2D_TICKS) * 10) / 10 for i in range(MAP_2D_TICKS + 1])


  def sample_event_plot (self, event, target, output, ax):
    EventVisualizer(event).density_map(show_truth=False, ax=ax)
    circle_width = JET_SIZE / (ETA_RANGE[1] - ETA_RANGE[0])
    circle_height = JET_SIZE / (PHI_RANGE[1] - PHI_RANGE[0])
    for i in range(0, len(target), 2):
      ax.add_patch(patches.Ellipse(Position(target[i], target[i+1]).relative(), circle_width, circle_height, color='red', fill=False))
    for i in range(0, len(output), 2):
      ax.add_patch(patches.Ellipse(Position(output[i], output[i+1]).relative(), circle_width, circle_height, color='blue', fill=False))

  def distances_histogram (self, starts, ends, ax):
    def distance (start, end):
      if np.sign(end[1]) != np.sign(start[1]):
        return np.linalg.norm([start[0] - end[0], 2 * np.pi - abs(start[1] - end[1])])
      else:
        return np.linalg.norm([start[0] - end[0], start[1] - end[1]])

    distances = [distance(start, end) for start, end in zip(starts, ends)]
    ax.hist(distances, bins=100)
    ax.set_xlabel('distance')
    ax.set_ylabel('count')

  def distances_by_pt_plot (self, starts, ends, events, ax):
    def distance (start, end):
      if np.sign(end[1]) != np.sign(start[1]):
        return np.linalg.norm([start[0] - end[0], 2 * np.pi - abs(start[1] - end[1])])
      else:
        return np.linalg.norm([start[0] - end[0], start[1] - end[1]])

    def pt (event):
      # sum of event.true_four_momentum().pt for all taus in the event
      return sum([momentum.p_t for momentum in event.true_four_momentum()])

    distances = [distance(start, end) for start, end in zip(starts, ends)]
    pts = [pt(event) for event in events]
    ax.scatter(pts, distances)
    ax.set_xlabel('pt')
    ax.set_ylabel('distance')