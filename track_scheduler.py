
import os
import signal
from datetime import datetime
import zmq
import json
import threading
from log import *
from track_worker import trackWorker

# table structure: pid, video, status, progress
class workerTable():
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.lock = threading.Lock()
        self.table = []

    def on_message(self, msg_dict):
        if msg_dict['status'] == '0':
            self.insert(msg_dict)
        elif msg_dict['status'] == '1':
            self.update(msg_dict)
        elif msg_dict['status'] == '2':
            self.delete(msg_dict)
        else:
            logger.warn('wrong status: %s' % (msg_dict['status']))

        self.dump()

    def insert(self, msg_dict):
        logger.debug("insert...")
        self.lock.acquire()
        self.table.append([msg_dict['pid'], msg_dict['video'], msg_dict['status'], msg_dict['progress']])
        self.lock.release()

    def update(self, msg_dict):
        logger.debug("update...")
        self.lock.acquire()
        for index in range(len(self.table)):
            if self.table[index][1] == msg_dict['video']:
                self.table[index][2] = msg_dict['status']
                self.table[index][3] = msg_dict['progress']
                break
        self.lock.release()

    def delete(self, msg_dict):
        logger.debug("delete...")
        self.lock.acquire()
        for index in range(len(self.table)):
            if self.table[index][1] == msg_dict['video']:
                del self.table[index]
                break
        self.lock.release()

    def dump(self):
        self.lock.acquire()
        # logger.debug("self.table: %s" % (self.table))
        with open(os.path.join(self.log_dir, 'status_table.txt'), 'at') as f:
            f.write('pid\tvideo\tstatus\tprogress\n')
            for index in range(len(self.table)):
                f.write('%s\n' % (self.table[index]))
            f.close()
        self.lock.release()

    def get_pid(self, msg_dict):
        for index in range(len(self.table)):
            if self.table[index][1] == msg_dict['video']:
                return int(self.table[index][0])
        return None

class statusThread(threading.Thread):
    def __init__(self, worker_table):
        threading.Thread.__init__(self)
        self.context = zmq.Context()
        # self.status_socket = self.context.socket(zmq.SUB)
        self.status_socket = self.context.socket(zmq.PULL)
        addr = 'tcp://*:6006'
        self.status_socket.bind(addr)
        # Subscribe on everything
        # self.status_socket.setsockopt_string(zmq.SUBSCRIBE, '')
        self.worker_table = worker_table

    def run(self):
        logger.info('status thread running...')
        while True:
            message = self.status_socket.recv()
            msg_dict = json.loads(message)
            if msg_dict['command'] == '3':
                logger.debug("recv status message: %s" % (message))
                self.worker_table.on_message(msg_dict)
            else:
                logger.warn('recv wrong status message: %s' % (message))

if __name__ == "__main__":
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser('./log'), subdir)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    set_logger(logger, log_dir)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    worker_table = workerTable(log_dir)

    status_thread = statusThread(worker_table)
    status_thread.start()
    logger.info('main thread running...')

    context = zmq.Context()
    scheduler = context.socket(zmq.REP)
    addr = 'tcp://*:6001'
    scheduler.bind(addr)
    while True:
        message = scheduler.recv()
        logger.debug("recv control message: %s" % (message))
        msg_dict = json.loads(message)
        # TODO: schedule, queue and wait
        if msg_dict['command'] == '0':
            pid = worker_table.get_pid(msg_dict)
            if pid is None:
                p = trackWorker(msg_dict['video'], msg_dict['model'])
                p.start()
            else:
                logger.warn("start video %s repeatedly" % (msg_dict['video']))
            msg_dict.update({'result': '0'})
            response = json.dumps(msg_dict)
            scheduler.send_string(response)

        elif msg_dict['command'] == '1':
            pid = worker_table.get_pid(msg_dict)
            if pid is not None:
                os.kill(pid, signal.SIGKILL)
                worker_table.delete(msg_dict)
                worker_table.dump()
                msg_dict.update({'result': '0'})
                response = json.dumps(msg_dict)
                scheduler.send_string(response)
            else:
                logger.warn('no pid processing video: %s' % (msg_dict['video']))
                msg_dict.update({'result': '0'})
                # msg_dict.update({'result': '1', 'info': 'not running'})
                response = json.dumps(msg_dict)
                scheduler.send_string(response)