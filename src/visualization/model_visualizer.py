import matplotlib.pyplot as plt
import numpy as np

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

  def plot_results (self, outputs, targets, output_file):
    print(f'Number of outputs that are all zeros: {len([1 for output in outputs if output[0] == 0 and output[1] == 0])}')
    print(f'Number of targets that are all zeros: {len([1 for target in targets if target[0] == 0 and target[1] == 0])}')
    
    # draw two plots side by side
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    self.arrows_on_eta_phi_plot(outputs, targets, axs[0], color='blue')
    self.distances_histogram(outputs, targets, axs[1])
    plt.savefig(output_file)
    plt.show()

  def arrows_on_eta_phi_plot (self, starts, ends, ax, **kwargs):
    def arrow_with_color (x, y, dx, dy, **kwargs):
      distance_normalized = min(1, max(0, 0.5 + 0.5 * np.linalg.norm([dx, dy]) / 2))
      color = (distance_normalized, 0, 1 - distance_normalized, 0.6)
      ax.arrow(x, y, dx, dy, head_width=0.1, head_length=0.1, fc=color, ec=color, **kwargs)

    for start, end in zip(starts, ends):
      arrow_with_color(start[1], start[0], end[1]-start[1], end[0] - start[0], **kwargs)

    ax.set_xlabel('phi')
    ax.set_ylabel('eta')
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-2.5, 2.5)

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
