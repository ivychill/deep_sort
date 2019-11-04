import os
import cv2
import numpy as np
from datetime import datetime
import json
import base64
import msgpack
import ffmpeg
from PIL import Image
from io import BytesIO
from time import sleep
import os
import signal
from log import logger

# CenterNet
# 20190821 使用更深的网络 hg105 速度有点慢，在212上0.7帧每秒。
'''
跑完视频可以生成特征的npy文件，但只包含特征，不包含位置
'''
import sys

CENTERNET_PATH = './CenterNet/src/lib/'
sys.path.insert(0, CENTERNET_PATH)
from detectors.detector_factory import detector_factory
from datasets.dataset_factory import dataset_factory
from opts import opts

from mot_track_kc import KCTracker
# from mot_track_simp import Sort as KCTracker
from util import COLORS_10, draw_bboxes, draw_bboxes_conf

import time
import warnings

warnings.filterwarnings("ignore", category=Warning)


def bbox_to_xywh_cls_conf(bbox_xyxyc, conf_thresh=0.5):
    if any(bbox_xyxyc[:, 4] >= conf_thresh):
        bbox = bbox_xyxyc[bbox_xyxyc[:, 4] >= conf_thresh, :]
        bbox[:, 2] = bbox[:, 2] - bbox[:, 0]  #
        bbox[:, 3] = bbox[:, 3] - bbox[:, 1]  #
        return bbox
    else:
        return []

def meargeResult(results, results_crop, y_top, x_left, crop_h, crop_w):
    # 大框结果中删除小框区域的结果，边界上的需要保留。也就是完全在小框内才删除。x>crop_left
    x_right = x_left + crop_w
    y_down = y_top + crop_h
    mask_inbox_x = (results[:, 0] > x_left) & (results[:, 2] < x_right)
    mask_inbox_y = (results[:, 1] > y_top) & (results[:, 3] < y_down)
    mask_big = np.logical_not(mask_inbox_x & mask_inbox_y)
    results_big = results[mask_big, :]
    # 小框内结果只要触及边界的需要删除
    mask_crop_x = (results_crop[:, 0] > 0) & (results_crop[:, 2] < crop_w)
    mask_crop_y = (results_crop[:, 1] >= 0) & (results_crop[:, 3] < crop_h)
    mask_small = mask_crop_x & mask_crop_y
    results_small = results_crop[mask_small, :]
    results_small[:, 0] += x_left
    results_small[:, 1] += y_top
    results_small[:, 2] += x_left
    results_small[:, 3] += y_top
    results_all = np.vstack((results_big, results_small))
    return results_all, results_big, results_small


def showResult(result_xyxyc, ori_im, save_pth, color=(0, 255, 0), conf_th=0.4):
    ori_im = ori_im.copy()
    result_xyxyc = result_xyxyc.copy()
    result_xyxyc = result_xyxyc[result_xyxyc[:, 4] > conf_th]
    for d in result_xyxyc:
        x, y, x2, y2, conf = d
        cv2.putText(ori_im, str(conf), (int(x), int(y + 15)), cv2.FONT_HERSHEY_PLAIN, 1,
                    color, 1)
        cv2.rectangle(ori_im, (int(x), int(y)), (int(x2), int(y2)), (0, 255, 0), 1)
    cv2.imwrite(save_pth, ori_im)


def makeFile(file_pth):
    file_dir, file_name = os.path.split(file_pth)
    os.makedirs(file_dir, exist_ok=True)
    if os.path.exists(file_pth):
        os.remove(file_pth)


def build_image_msg(camera, ori_im, identities, bboxes):
    scale = 3.0/4
    # img_rgb = cv2.cvtColor(ori_im, cv2.COLOR_RGB2BGR)
    height, width = ori_im.shape[0:2]
    img_lr = cv2.resize(ori_im, (int(width*scale), int(height*scale)))
    time_1 = time.time()

    img = Image.fromarray(img_lr)
    output_buffer = BytesIO()
    img.save(output_buffer, format='JPEG')
    binary_data = output_buffer.getvalue()
    time_2 = time.time()

    base64_data = base64.b64encode(binary_data)
    time_3 = time.time()

    tracks = []
    for i, d in enumerate(bboxes):
        track = {'id': str(int(identities[i])), 'x': str(int(d[0]*scale)), 'y': str(int(d[1]*scale)), 'w': str(int((d[2]-d[0])*scale)), 'h': str(int((d[3]-d[1])*scale))}
        tracks.append(track)
    # logger.debug("tracks: %s" % (tracks))
    msg_dict = {'camera':camera, 'command':'2', 'image':base64_data.decode('utf-8'), 'track':tracks}
    message = json.dumps(msg_dict)
    time_4 = time.time()

    # logger.debug('build_image_msg: 1.encode jpg: {} s, 2.base64: {} s, 3.dumps: {} s'.format(time_2-time_1, time_3-time_2, time_4-time_3))
    return message


class Detector(object):
    def __init__(self, opt, min_confidence=0.4, max_cosine_distance=0.2, max_iou_distance=0.7, max_age=30,
                 out_dir='res/'):
        self.opt = opt
        self.vdo = cv2.VideoCapture()
        self.out_dir = out_dir
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            # os.makedirs(out_dir + '/imgs', exist_ok=True)

        # centerNet detector
        self.detector = detector_factory[self.opt.task](self.opt)
        # self.crop_detector = detector_factory[self.opt.task](crop_opt)

        self.kc_tracker = KCTracker(confidence_l=0.2, confidence_h=0.4,use_filter=True, max_cosine_distance=max_cosine_distance,
                                    max_iou_distance=max_iou_distance, max_age=max_age)
        # self.kc_tracker = KCTracker()

        _, filename = os.path.split(self.opt.vid_path)

        model_name = '_centernet_'
        timestamp = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')

        self.mot_txt = os.path.join(self.out_dir, filename[:-4] + model_name + timestamp + '_ini.txt')
        self.mot_txt_filter = os.path.join(self.out_dir, filename[:-4] + model_name + timestamp + '.txt')
        self.mot_txt_bk = os.path.join(self.out_dir, filename[:-4] + model_name + timestamp + '_bk.txt')
        self.det_txt = os.path.join(self.out_dir, filename[:-4] + model_name + timestamp + '_det.txt')
        makeFile(self.mot_txt)
        makeFile(self.mot_txt_bk)
        makeFile(self.mot_txt_filter)
        makeFile(self.det_txt)
        self.video_name = os.path.join(self.out_dir, filename[:-4] + model_name + timestamp + '.webm')
        self.features_npy = os.path.join(self.out_dir, filename[:-4] + model_name + timestamp + '_det.npy')
        self.save_feature = False
        self.all_features = []
        self.write_det_txt = False
        self.write_video = False
        self.use_tracker = True
        self.person_id = 1
        self.write_img = False
        self.write_bk = False
        self.temp_dir = filename[:-4]
        if self.write_img:
            self.img_dir = os.path.join(self.out_dir + '/' + self.temp_dir, 'imgs')
            os.makedirs(self.img_dir, exist_ok=True)

    def open(self, video_path):
        if self.opt.input_type == 'webcam':
            self.vdo.open(self.opt.webcam_ind)
        elif self.opt.input_type == 'ipcam':
            self.vdo.open(self.opt.ipcam_url)
        else:  # video
            assert os.path.isfile(self.opt.vid_path), "Error: path error"
            self.vdo.open(self.opt.vid_path)

        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))     # 1920
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))   # 1080
        self.fps = self.vdo.get(cv2.CAP_PROP_FPS)
        self.frame_count =int(self.vdo.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.debug('self.im_width: {}, self.im_height: {}'.format(self.im_width, self.im_height))
        if self.opt.input_type == 'ipcam':
            self.process = (
                ffmpeg
                    .input(self.opt.ipcam_url, ss=0, stimeout=str(2*1000000), rtsp_transport='tcp', vcodec='h264_cuvid')
                    .output('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(self.im_width, self.im_height))
                    .run_async(pipe_stdout=True, pipe_stderr=True)
            )

        self.area = 0, 0, self.im_width, self.im_height
        if self.write_video:
            # fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fourcc = cv2.VideoWriter_fourcc(*'VP80')
            self.output = cv2.VideoWriter(self.video_name, fourcc, self.fps, (self.im_width, self.im_height))

    def saveFeature(self, file_name, np_mat):
        if np_mat is not None:
            np.save(file_name, np_mat)
            print('save npy file:{},size:{}'.format(file_name, np_mat.shape))
        else:
            print('empty! save nothing!')

    def detect(self, video=None, camera=None, pid=None, socket_web=None, socket_scheduler=None):
        xmin, ymin, xmax, ymax = self.area
        frame_no = 0
        avg_fps = 0.0
        # crop_top, crop_left, crop_h, crop_w = 0, 650, 150, 520
        # ret, ori_im = self.vdo.read()
        # if (camera is not None) and (not ret):
        #     logger.warn("read from camera %s fail" % (camera))
        ori_im = self.process.stdout.read(self.im_width * self.im_height * 3)
        self.process.stdout.flush()
        if len(ori_im) <= 0:
            logger.warn("read from camera %s fail" % (camera))

        # while ret:
        while len(ori_im) > 0:
            ori_im = np.frombuffer(ori_im, dtype='uint8')
            ori_im = ori_im.reshape(self.im_height, self.im_width, 3)
            frame_no += 1
            start = time.time()
            im = ori_im[ymin:ymax, xmin:xmax]
            try:
                results_big = self.detector.run(im)['results'][self.person_id]
                results = results_big
            except Exception as e:
                logger.warn("Exception: {}".format(e))
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            if self.write_det_txt:
                bbox_xyxyc = results
                if len(results) > 0:
                    bbox_xyxyc = bbox_xyxyc[bbox_xyxyc[:, 4] > 0.05]
                    for i, d in enumerate(bbox_xyxyc):
                        with open(self.det_txt, 'a') as f:
                            msg = '%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f\n' % (
                                frame_no, -1, d[0], d[1], d[2] - d[0], d[3] - d[1], d[4])
                            f.write(msg)

            if len(results) > 0:
                bbox_xywhcs = bbox_to_xywh_cls_conf(results, conf_thresh=0.05)
            else:
                bbox_xywhcs = []

            track_start = time.time()

            if len(bbox_xywhcs) > 0:
                if self.use_tracker:
                    outputs, features = self.kc_tracker.update(frame_no, bbox_xywhcs, im)
                    if self.save_feature:
                        if self.all_features is None:
                            self.all_features = features
                        else:
                            self.all_features = np.vstack((self.all_features, features))

                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:, 1:5]
                        identities = outputs[:, 0]
                        confs = outputs[:, -1]
                        if camera is not None:
                            message = build_image_msg(camera, ori_im, identities, bbox_xyxy)
                            time_before = time.time()
                            socket_web.send_string(message)
                            # logger.debug("send image message")
                            time_after = time.time()
                            # logger.debug('zmq send cost {} s'.format(time_after - time_before))

                        ori_im = draw_bboxes_conf(ori_im, bbox_xyxy, confs, identities, offset=(xmin, ymin))
                        for i, d in enumerate(bbox_xyxy):
                            with open(self.mot_txt, 'a') as f:
                                msg = '%d,%d,%.2f,%.2f,%.2f,%.2f\n' % (
                                    frame_no, identities[i], d[0], d[1], d[2] - d[0], d[3] - d[1])
                                f.write(msg)
                            if self.write_bk:
                                with open(self.mot_txt_bk, 'a') as f:
                                    msg = '%d,%d,%.2f,%.2f,%.2f,%.2f,%.3f\n' % (
                                        frame_no, identities[i], d[0], d[1], d[2] - d[0], d[3] - d[1], confs[i])
                                    f.write(msg)
                    else:
                        logger.debug("no id")
                        if camera is not None:
                            message = build_image_msg(camera, ori_im, [], [])
                            socket_web.send_string(message)
                            logger.debug("send image message")

            else:
                self.kc_tracker.update(frame_no, bbox_xywhcs, im)
                logger.debug("no id")
                if camera is not None:
                    message = build_image_msg(camera, ori_im, [], [])
                    socket_web.send_string(message)
                    logger.debug("send image message")
            end = time.time()
            fps = 1 / (end - start)
            avg_fps += fps
            if frame_no % 100 == 0:
                logger.debug("frame_no {} detect cost {}s, fps {}, track cost {}s".format(frame_no, end-start, fps, end-track_start))
                if video is not None:
                    progress = round(float(frame_no)/self.frame_count, 2)
                    msg_dict = {'command': '3', 'video': video, 'status': '1', 'progress': str(progress), 'pid': str(pid)}
                    message = json.dumps(msg_dict)
                    socket_web.send_string(message)
                    socket_scheduler.send_string(message)
                    logger.debug("send status message: %s" % (message))

            if self.write_video:
                self.output.write(ori_im)
            if self.write_img:
                cv2.imwrite(os.path.join(self.img_dir, '{:06d}.jpg'.format(frame_no)), ori_im)

            # try:
            #     ret, ori_im = self.vdo.read()
            # except Exception as e:
            #     logger.warn("Exception: {}".format(e))
            # if (camera is not None) and (not ret):
            #     logger.warn("read frame: %d from camera %s fail" % (frame_no, camera))
            #     self.vdo.release()
            #     self.vdo.open(self.opt.ipcam_url)
            #     ret, ori_im = self.vdo.read()

            time_before = time.time()
            # send 1 out of 2
            self.process.stdout.read(self.im_width * self.im_height * 3)
            ori_im = self.process.stdout.read(self.im_width * self.im_height * 3)
            self.process.stdout.flush()
            time_after = time.time()
            # logger.debug('read a frame from camera cost {} s'.format(time_after - time_before))
            index = 0
            while len(ori_im) <= 0:
                logger.warn("read frame: %d from camera %s fail No. %d" % (frame_no, camera, index))
                os.kill(self.process.pid, signal.SIGTERM)
                logger.warn("restart ffmpeg")
                time_before = time.time()
                self.process = (
                    ffmpeg
                        .input(self.opt.ipcam_url, ss=0, stimeout=str(2*1000000), rtsp_transport='tcp', vcodec='h264_cuvid')
                        .output('pipe:', format='rawvideo', pix_fmt='rgb24',
                                s='{}x{}'.format(self.im_width, self.im_height))
                        .run_async(pipe_stdout=True, pipe_stderr=True)
                )
                time_after = time.time()
                logger.debug('restart ffmpeg cost {} s'.format(time_after - time_before))
                self.process.stdout.read(self.im_width * self.im_height * 3)
                ori_im = self.process.stdout.read(self.im_width * self.im_height * 3)
                index += 1

        if self.opt.input_type == 'ipcam':
            os.kill(self.process.pid, signal.SIGTERM)
        else:
            self.vdo.release()

        if self.save_feature:
            self.saveFeature(self.features_npy, self.all_features)

def process(video, pid, socket_web, socket_scheduler):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # MODEL_PATH = './CenterNet/models/centernet_coco_hg_model_best0917.pth'  # 0917
    # ARCH = 'hourglass'
    # EXP_ID = 'coco_hg'
    MODEL_PATH = './CenterNet/models/dla_best1012.pth'
    ARCH = 'dla_34'
    EXP_ID = 'pascal_dla_512'
    TASK = 'ctdet'  # or 'multi_pose' for human pose estimation
    # opt = opts().init(
    #     '{} --load_model {} --arch {} --exp_id {} --input_res 1024 --resume --flip_test --test_scales 0.75,1,1.25'.format(
    #         TASK, MODEL_PATH, ARCH, EXP_ID).split(' '))
    opt = opts().init(
        '{} --load_model {} --arch {} --exp_id {} --input_res 512 --resume --test_scales 1'.format(
            TASK, MODEL_PATH, ARCH, EXP_ID).split(' '))
    Dataset = dataset_factory['kc']
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    opt.input_type = 'vid'  # for video
    opt.vis_thresh = 0.4
    path_dir = './videos'
    filename = os.path.join(path_dir, video)
    start_time = datetime.now()
    print('start time:', start_time)
    opt.vid_path = filename  #
    det = Detector(opt, min_confidence=opt.vis_thresh, max_cosine_distance=0.2,
                   max_iou_distance=0.7, max_age=30, out_dir='videos/results')
    det.save_feature = False
    det.write_det_txt = False
    det.use_tracker = True
    det.write_video = True
    det.write_bk = False
    print('################### start :', filename)
    det.open(filename)
    det.detect(video=video, pid=pid, socket_web=socket_web, socket_scheduler=socket_scheduler)
    det.kc_tracker.saveResult(det.mot_txt_filter)
    end_time = datetime.now()
    print('################### finish :', filename, end_time)
    print(' cost hour:', (end_time - start_time) )

def process_rt(camera, pid, socket_web, socket_scheduler):
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    MODEL_PATH = './CenterNet/models/dla_best1012.pth'
    ARCH = 'dla_34'
    EXP_ID = 'pascal_dla_512'
    TASK = 'ctdet'  # or 'multi_pose' for human pose estimation
    opt = opts().init(
        '{} --load_model {} --arch {} --exp_id {} --input_res 512 --resume --test_scales 1'.format(
            TASK, MODEL_PATH, ARCH, EXP_ID).split(' '))
    Dataset = dataset_factory['kc']
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    opt.input_type = 'ipcam'  # for camera
    # TODO: camera
    opt.ipcam_url = "rtsp://admin:abcd1234@" + camera + ":554"
    opt.vis_thresh = 0.4
    path_dir = './videos'
    filename = os.path.join(path_dir, camera)
    opt.vid_path = filename
    start_time = datetime.now()
    print('start time:', start_time)
    det = Detector(opt, min_confidence=opt.vis_thresh, max_cosine_distance=0.2,
                   max_iou_distance=0.7, max_age=30, out_dir='videos/results')
    det.save_feature = False
    det.write_det_txt = False
    det.use_tracker = True
    det.write_video = True
    det.write_bk = False
    print('################### start :', camera)
    det.open(filename)
    det.detect(camera=camera, pid=pid, socket_web=socket_web, socket_scheduler=socket_scheduler)
    # det.kc_tracker.saveResult(det.mot_txt_filter)
    end_time = datetime.now()
    print('################### finish :', camera, end_time)
    print(' cost hour:', (end_time - start_time) )

if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    ## model choose

    # MODEL_PATH = './CenterNet/models/ctdet_dla_zj.pth'  # 20190917
    # ARCH = 'dla_34'
    # TASK = 'ctdet'  # or 'multi_pose' for human pose estimation
    # opt = opts().init(
    #     '{} --load_model {} --arch {} --input_res 800 --test_scales 0.75,1'.format(TASK, MODEL_PATH, ARCH).split(
    #         ' '))
    # Dataset = dataset_factory['kc']
    # opt = opts().update_dataset_info_and_set_heads(opt, Dataset)

    # # best
    # MODEL_PATH = './CenterNet/models/ctdet_coco_hg_zjnew.pth'
    # ARCH = 'hourglass'
    # EXP_ID = 'coco_hg'
    # TASK = 'ctdet'  # or 'multi_pose' for human pose estimation
    # opt = opts().init(
    #     '{} --load_model {} --arch {} --exp_id {} --input_res 1024 --resume --flip_test --test_scales 0.75,1,1.25'.format(
    #         TASK, MODEL_PATH, ARCH, EXP_ID).split(' '))

    # global crop_opt
    # crop_MODEL_PATH = './CenterNet/models/ctdet_dla_for_crop_0927.pth'
    # crop_ARCH = 'dla_34'
    # EXP_ID = 'coco_hg'
    # TASK = 'ctdet'  # or 'multi_pose' for human pose estimation
    # crop_opt = opts().init(
    #     '{} --load_model {} --arch {} --exp_id {} --input_res 1024 --resume --flip_test --test_scales 0.75,1,1.25'.format(
    #         TASK, crop_MODEL_PATH, crop_ARCH, EXP_ID).split(' '))
    # crop_Dataset = dataset_factory['kc']
    # crop_opt = opts().update_dataset_info_and_set_heads(crop_opt, crop_Dataset)

    # #  0917
    MODEL_PATH = './CenterNet/models/centernet_coco_hg_model_best0917.pth'  # 0917
    ARCH = 'hourglass'
    EXP_ID = 'coco_hg'
    TASK = 'ctdet'  # or 'multi_pose' for human pose estimation
    opt = opts().init(
        '{} --load_model {} --arch {} --exp_id {} --input_res 1024 --resume --flip_test --test_scales 0.75,1,1.25'.format(
            TASK, MODEL_PATH, ARCH, EXP_ID).split(' '))
    Dataset = dataset_factory['kc']
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)

    opt.input_type = 'vid'  # for video
    opt.vis_thresh = 0.4

    # path_dir = '/home/kcadmin/user/Dataset/level2_vedio'
    path_dir = './videos'
    video_list = os.listdir(path_dir)
    seq = ['b1.mp4','b3.mp4','b5.mp4']
    video_file_list = [os.path.join(path_dir, vl) for vl in seq]
    # video_file_list = ['/home/kcadmin/user/Dataset/level2_vedio/b1.mp4','/home/kcadmin/user/Dataset/level2_vedio/b3.mp4','/home/kcadmin/user/Dataset/level2_vedio/b5.mp4']
    video_file_list.sort(reverse=False)
    print(video_file_list)
    start_time = datetime.now()
    print('start time:', start_time)
    for filename in video_file_list:
        opt.vid_path = filename  #
        det = Detector(opt, min_confidence=opt.vis_thresh, max_cosine_distance=0.2,
                       max_iou_distance=0.7, max_age=30, out_dir='results/res_20191007')
        det.save_feature = False
        det.write_det_txt = True
        det.use_tracker = True
        det.write_video = True
        det.write_bk = True
        print('################### start :', filename)
        det.open(filename)
        det.detect()
        det.kc_tracker.saveResult(det.mot_txt_filter)
        end_time = datetime.now()
        print('################### finish :', filename, end_time)
        print(' cost hour:', (end_time - start_time) )
