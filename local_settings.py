import pathlib2

# ----------------- DON'T TOUCH ANYTHING ABOVE THIS LINE ----------------------




# These paths are relative to the main project directory
MODEL_DIR = 'cnn_tf_model'
TRAINING_DATA_DIR = 'training_data'




# ----------------- DON'T TOUCH ANYTHING BELOW THIS LINE ----------------------

PROJECT_DIR = pathlib2.Path(__file__).absolute().parent

MODEL_DIR = PROJECT_DIR / MODEL_DIR
TRAINING_DATA_DIR = PROJECT_DIR / TRAINING_DATA_DIR
