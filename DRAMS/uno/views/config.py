# import the necessary packages
import os

# Base directory (project root, one level up from DRAMS)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# initialize the class labels in the dataset
CLASSES = ["Flood", "Fire"]

# set the path to the serialized model after training
MODEL_PATH = os.path.join(BASE_DIR, '..', 'output', 'flood.h5')

# Fire model path
FIRE_MODEL_PATH = os.path.join(BASE_DIR, '..', 'output', 'fire.h5')

# define training hyperparameters (kept for reference)
TRAIN_SPLIT = 0.75
VAL_SPLIT = 0.1
TEST_SPLIT = 0.25
MIN_LR = 2e-7
MAX_LR = 2e-5
BATCH_SIZE = 4
STEP_SIZE = 8
CLR_METHOD = "triangular"
NUM_EPOCHS = 40
