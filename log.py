#

import logging.handlers
import os

LOG_PATH = './logs'
MAX_LOG_SIZE = 2560000
LOG_BACKUP_NUM = 4000


def set_logger(logger, log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, 'track.log')
    handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=MAX_LOG_SIZE, backupCount=LOG_BACKUP_NUM)
    formatter = logging.Formatter('%(asctime)s %(process)d %(processName)s %(threadName)s %(filename)s %(lineno)d %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)
    logger.setLevel(logging.DEBUG)

logger = logging.getLogger('track')
