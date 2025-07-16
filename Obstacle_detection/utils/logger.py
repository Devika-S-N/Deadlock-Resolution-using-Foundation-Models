import logging
from datetime import datetime
import os

def setup_logger(log_dir=None):
    """Configure and return a logger instance"""
    if log_dir is None:
        log_dir = f"logs/logs_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger('path_planner')
    logger.setLevel(logging.DEBUG)
    
    # Create file handler
    log_file = os.path.join(log_dir, 'path_planning.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    
    return logger, log_dir