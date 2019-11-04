import time,os
from multiprocessing import Process
import zmq
import json
import time
from log import logger
import main_cascade
# import main_detector
import main_detector_xz as main_detector

class trackWorkerRt(Process):
    def __init__(self, camera):
        super(trackWorkerRt, self).__init__()
        logger.info("track worker started...")
        self.camera = camera

    def run(self):
        logger.info("track worker running...")
        self.context = zmq.Context()

        # self.socket_web = self.context.socket(zmq.PUB)
        self.socket_web = self.context.socket(zmq.PUSH)
        self.addr_web = 'tcp://127.0.0.1:8004'
        self.socket_web.setsockopt(zmq.SNDHWM, 1)
        self.socket_web.connect(self.addr_web)

        # self.socket_scheduler = self.context.socket(zmq.PUB)
        self.socket_scheduler = self.context.socket(zmq.PUSH)
        self.addr_scheduler = 'tcp://127.0.0.1:8006'
        self.socket_scheduler.connect(self.addr_scheduler)

        self.process_id = os.getpid()
        msg_dict = {'command':'3', 'camera': self.camera, 'status': '0', 'pid': str(self.process_id)}
        message = json.dumps(msg_dict)
        self.socket_scheduler.send_string(message)
        logger.debug("send status message: %s" % (message))

        main_detector.process_rt(self.camera, self.process_id, self.socket_web, self.socket_scheduler)

        msg_dict = {'command': '3', 'camera': self.camera, 'status': '1', 'pid': str(self.process_id)}
        message = json.dumps(msg_dict)
        self.socket_scheduler.send_string(message)
        logger.info('send finish message: %s' % (message))
