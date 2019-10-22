
from datetime import datetime
import logging.handlers
import os
import zmq
import time
import json
import threading

LOG_PATH = './logs'
MAX_LOG_SIZE = 2560000
LOG_BACKUP_NUM = 4000

class statusThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.context = zmq.Context()
        # self.status_socket = self.context.socket(zmq.SUB)
        self.status_socket = self.context.socket(zmq.PULL)
        addr = 'tcp://*:7004'
        self.status_socket.bind(addr)
        # Subscribe on everything
        # self.status_socket.setsockopt_string(zmq.SUBSCRIBE, '')

    def run(self):
        logger.info('status thread running...')
        while True:
            message = self.status_socket.recv()
            msg_dict = json.loads(message)
            if msg_dict['command'] == '2':
                logger.debug("recv image message")
            else:
                logger.warn('recv wrong status message: %s' % (message))

def set_logger(logger, log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, 'web.log')
    handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=MAX_LOG_SIZE, backupCount=LOG_BACKUP_NUM)
    formatter = logging.Formatter('%(asctime)s %(process)d %(processName)s %(threadName)s %(filename)s %(lineno)d %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)
    logger.setLevel(logging.DEBUG)

logger = logging.getLogger('web')

subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
log_dir = os.path.join(os.path.expanduser('./log/web'), subdir)
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
set_logger(logger, log_dir)

status_thread = statusThread()
status_thread.start()
logger.info('main thread running...')

context = zmq.Context()
cmd_socket = context.socket(zmq.REQ)
addr = 'tcp://127.0.0.1:7001'
cmd_socket.connect(addr)

time.sleep(5)

msg_dict = {'command': '0', 'camera': '10.15.10.4'}
message = json.dumps(msg_dict)
cmd_socket.send_string(message)
response = cmd_socket.recv()
logger.debug("receive message: %s" % (response))

time.sleep(30)

msg_dict = {'command': '1', 'camera': '10.15.10.2'}
message = json.dumps(msg_dict)
cmd_socket.send_string(message)
response = cmd_socket.recv()
logger.debug("receive message: %s" % (response))