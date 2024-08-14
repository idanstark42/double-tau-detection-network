from math import pi

# math
ETA_RANGE = (-2.5, 2.5)
PHI_RANGE = (-pi, pi)

# Visualization
MAP_2D_TICKS = 5
HISTOGRAM_BINS = 50

# IO
DATA_DIR = 'data'
MODELS_DIR = 'models'

# Data
RESOLUTION = 100
DATASET_FIELDS = ['event', 'clusters', 'tracks', 'truthTaus']
DATA_FILE = 'ggXtautau_mX20_run3year1_x10'

# TRAINING
EPOCHS = 100
BATCH_SIZE = 256

TRAINING_PERCENTAGE = 0.7
VALIDATION_PERCENTAGE = 0.2
TEST_ARROWS_PERCENTAGE = 0.1    
