import time,os
from multiprocessing import Process
import zmq
import json
import time
from log import logger
import main_cascade
import main_detector

class trackWorker(Process):
    def __init__(self, video, model):
        super(trackWorker, self).__init__()
        logger.info("track worker started...")
        self.video = video
        self.model = model

    def run(self):
        logger.info("track worker running...")
        self.context = zmq.Context()

        # self.socket_web = self.context.socket(zmq.PUB)
        self.socket_web = self.context.socket(zmq.PUSH)
        self.addr_web = 'tcp://127.0.0.1:6004'
        # self.addr_web = 'tcp://10.15.10.67:6004'
        self.socket_web.connect(self.addr_web)

        # self.socket_scheduler = self.context.socket(zmq.PUB)
        self.socket_scheduler = self.context.socket(zmq.PUSH)
        self.addr_scheduler = 'tcp://127.0.0.1:6006'
        self.socket_scheduler.connect(self.addr_scheduler)

        self.process_id = os.getpid()
        msg_dict = {'command':'3', 'video': self.video, 'status': '0', 'progress': '0', 'pid': str(self.process_id)}
        message = json.dumps(msg_dict)
        # don't send this message to web server
        # self.socket_web.send_string(message)
        self.socket_scheduler.send_string(message)
        logger.debug("send status message: %s" % (message))

        if self.model == 'centernet':
            main_detector.process(self.video, self.process_id, self.socket_web, self.socket_scheduler)
            msg_dict = {'command': '3', 'video': self.video, 'status': '2', 'progress': '1', 'pid': str(self.process_id)}
            message = json.dumps(msg_dict)
            self.socket_web.send_string(message)
            self.socket_scheduler.send_string(message)
            logger.info('send finish message: %s' % (message))
        elif self.model == 'cascade':
            main_cascade.process(self.video, self.process_id, self.socket_web, self.socket_scheduler)
            msg_dict = {'command': '3', 'video': self.video, 'status': '2', 'progress': '1', 'pid': str(self.process_id)}
            message = json.dumps(msg_dict)
            self.socket_web.send_string(message)
            self.socket_scheduler.send_string(message)
            logger.info('send finish message: %s' % (message))
        else:
            logger.warn('unknown model: %s' % (self.model))