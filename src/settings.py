from math import pi

# math
ETA_RANGE = (-2.5, 2.5)
PHI_RANGE = (-pi, pi)
JET_SIZE = 0.2

# Visualization
MAP_2D_TICKS = 5
HISTOGRAM_BINS = 50

# IO
DATA_DIR = 'data'
MODELS_DIR = 'models'

# Data
RESOLUTION = 128
DATASET_FIELDS = ['event', 'clusters', 'tracks', 'truthTaus']
DATA_FILE = 'merged_x3'

CHANNEL_START = 525007
FIELDS_TO_NORMALIZE = {
  'clusters': [], #['center_mag', 'center_lambda', 'second_r', 'second_lambda'],
  'tracks': [] #['number_of_pixel_hits', 'number_of_sct_hits', 'number_of_trt_hits', 'q_over_p'],
}

# TRAINING
EPOCHS = 2
BATCH_SIZE = 64

TRAINING_PERCENTAGE = 0.7
VALIDATION_PERCENTAGE = 0.2
ARROWS_NUMBER = 1000
