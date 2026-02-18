import logging
from tensorflow.keras import backend as K

def setup_logger(log_file):
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def squeeze_last_dim(x):
    """Custom Lambda function to squeeze the last dimension."""
    return K.squeeze(x, axis=-1)

def mean_over_time(x):
    """Custom Lambda function to compute mean over time."""
    return K.mean(x, axis=1)

def expand_last_dim(x):
    """Custom Lambda function to expand the last dimension."""
    return K.expand_dims(x, axis=-1)

def weighted_multiplication(inputs):
    """Custom Lambda function for weighted multiplication."""
    return inputs[0] * inputs[1]