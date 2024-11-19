from math import pi

# math
ETA_RANGE = (-2.5, 2.5)
PHI_RANGE = (-pi, pi)
JET_SIZE = 0.2

# Visualization
MAP_2D_TICKS = 5
HISTOGRAM_BINS = 50
MAX_HISTOGRAM_SIZE=100000

# IO
DATA_DIR = 'data'
MODELS_DIR = 'models'
GRAPHS_DIR = 'plots'

# Data
RESOLUTION = 128
DATASET_FIELDS = ['event', 'clusters', 'tracks', 'truthTaus']
DATA_FILE = 'ggXtautau_mX20_run3year1_x10_x10'

CHANNEL_START = 525007
FIELDS_TO_NORMALIZE = {
  'clusters': [],
  'tracks': []
}

# TRAINING
EPOCHS = 100
BATCH_SIZE = 256

TRAINING_PERCENTAGE = 0.7
VALIDATION_PERCENTAGE = 0.2
ARROWS_NUMBER = 1000
