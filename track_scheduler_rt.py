
import os
import signal
from datetime import datetime
import zmq
import json
import threading
from log import *
from track_worker_rt import trackWorkerRt

# table structure: pid, camera, status
class workerTable():
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.lock = threading.Lock()
        self.table = []

    def on_message(self, msg_dict):
        if msg_dict['status'] == '0':
            self.insert(msg_dict)
        elif msg_dict['status'] == '1':
            self.delete(msg_dict)
        else:
            logger.warn('wrong status: %s' % (msg_dict['status']))

        self.dump()

    def insert(self, msg_dict):
        logger.debug("insert...")
        self.lock.acquire()
        self.table.append([msg_dict['pid'], msg_dict['camera'], msg_dict['status']])
        self.lock.release()

    def delete(self, msg_dict):
        logger.debug("delete...")
        self.lock.acquire()
        for index in range(len(self.table)):
            if self.table[index][1] == msg_dict['camera']:
                del self.table[index]
                break
        self.lock.release()

    def dump(self):
        self.lock.acquire()
        # logger.debug("self.table: %s" % (self.table))
        with open(os.path.join(self.log_dir, 'status_table_rt.txt'), 'at') as f:
            f.write('pid\tcamera\tstatus\n')
            for index in range(len(self.table)):
                f.write('%s\n' % (self.table[index]))
            f.close()
        self.lock.release()

    def get_pid(self, msg_dict):
        for index in range(len(self.table)):
            if self.table[index][1] == msg_dict['camera']:
                return int(self.table[index][0])
        return None

    def get_camera_num(self):
        return len(self.table)

    def get_camera(self):
        if len(self.table) >= 1:
            return self.table[0][1]
        else:
            return None

class statusThread(threading.Thread):
    def __init__(self, worker_table):
        threading.Thread.__init__(self)
        self.context = zmq.Context()
        # self.status_socket = self.context.socket(zmq.SUB)
        self.status_socket = self.context.socket(zmq.PULL)
        addr = 'tcp://*:7006'
        self.status_socket.bind(addr)
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

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    push_url = 'rtmp://10.15.10.24:1935/live/show.stream'

    worker_table = workerTable(log_dir)

    status_thread = statusThread(worker_table)
    status_thread.start()
    logger.info('main thread running...')

    context = zmq.Context()
    scheduler = context.socket(zmq.REP)
    addr = 'tcp://*:7001'
    scheduler.bind(addr)
    while True:
        message = scheduler.recv()
        logger.debug("recv control message: %s" % (message))
        msg_dict = json.loads(message)
        # TODO: schedule, queue and wait
        if msg_dict['command'] == '0':
            if worker_table.get_camera_num() >= 1:
                msg_dict.update({'result': '1', 'info': worker_table.get_camera()})
            else:
                pid = worker_table.get_pid(msg_dict)
                if pid is None:
                    p = trackWorkerRt(msg_dict['camera'], msg_dict['stat'], push_url)
                    p.start()
                else:
                    logger.warn("start camera %s repeatedly" % (msg_dict['camera']))
                msg_dict.update({'result': '0', 'url': push_url})
            response = json.dumps(msg_dict)
            scheduler.send_string(response)

        elif msg_dict['command'] == '1':
            pid = worker_table.get_pid(msg_dict)
            if pid is not None:
                os.kill(pid, signal.SIGKILL)
                worker_table.delete(msg_dict)
                worker_table.dump()
            else:
                logger.warn('no pid processing camera: %s' % (msg_dict['camera']))

            msg_dict.update({'result': '0', 'url': push_url})
            response = json.dumps(msg_dict)
            scheduler.send_string(response)