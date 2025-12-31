import os
import sys
import logging
logging.getLogger('fontTools.subset').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
def setup_logging(log_file_path):
    """Configure logging to file and stdout."""
    if os.path.exists(log_file_path) and log_file_path[-4:]=='.log':
        os.remove(log_file_path)
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Clear existing handlers
    logger.handlers.clear()
    # Set log format
    formatter = logging.Formatter()
    # File handler
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    # add handler to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
