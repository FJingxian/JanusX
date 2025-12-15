import os
import sys
import logging
logging.getLogger('fontTools.subset').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
def setup_logging(log_file_path):
    """set logging"""
    if os.path.exists(log_file_path) and log_file_path[-4:]=='.log':
        os.remove(log_file_path)
    # creart logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # clean exist handler
    logger.handlers.clear()
    # set log format
    formatter = logging.Formatter()
    # file handler
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    # handler of control panel
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    # add handler to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger