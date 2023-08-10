import logging
import os

def create_log(log_dir,
               format='[%(asctime)s] ==> %(message)s',
               datefmt='%Y-%m-%d / %H:%M'):

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # os.chdir(log_dir)

    # Create logger instance
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create log format for output
    formatter = logging.Formatter(format, datefmt)

    # File Handler
    log_file = os.path.join(log_dir, 'log.txt')
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # Console Handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    # add handler into logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

