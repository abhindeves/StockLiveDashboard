import logging
import os

# Configure the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s - %(message)s')  # Added filename to the format
logger = logging.getLogger(os.path.basename(__file__))

def get_logger():
    return logger