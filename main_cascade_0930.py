# -*-coding:utf-8-*-
import os, sys
import cv2
import numpy as np
import sys
from Cascade.cascade_detector import CascadeDetector
from datetime import datetime

from mot_track_kc import KCTracker
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


def makeFile(file_pth):
    file_dir, file_name = os.path.split(file_pth)
    os.makedirs(file_dir, exist_ok=True)
    if os.path.exists(file_pth):
        os.remove(file_pth)


class Detector(object):
    def __init__(self, vid_path, max_cosine_distance=0.2, max_iou_distance=0.7, max_age=30,
                 out_dir='res/'):
        self.vdo = cv2.VideoCapture()
        self.out_dir = out_dir
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        self.kc_tracker = KCTracker(confidence_l=0.5, confidence_h=0.8, max_cosine_distance=max_cosine_distance,
                                    max_iou_distance=max_iou_distance, max_age=max_age)

        _, filename = os.path.split(vid_path)
        self.mot_txt = os.path.join(self.out_dir, filename[:-4] + '_ini.txt')
        self.mot_txt_filter = os.path.join(self.out_dir, filename[:-4] + '.txt')
        self.mot_txt_bk = os.path.join(self.out_dir, filename[:-4] + '_bk.txt')
        self.det_txt = os.path.join(self.out_dir, filename[:-4] + '_det.txt')
        makeFile(self.mot_txt)
        makeFile(self.mot_txt_bk)
        makeFile(self.mot_txt_filter)
        makeFile(self.det_txt)
        self.video_name = os.path.join(self.out_dir, filename[:-4] + '_res.avi')
        self.features_npy = os.path.join(self.out_dir, filename[:-4] + '_det.npy')
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

    def open(self, video_path, input_type='vid'):
        assert os.path.isfile(video_path), "Error: path error"
        self.vdo.open(video_path)

        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.vdo.get(cv2.CAP_PROP_FPS)

        self.area = 0, 0, self.im_width, self.im_height
        if self.write_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.output = cv2.VideoWriter(self.video_name, fourcc, self.fps, (self.im_width, self.im_height))

    def saveFeature(self, file_name, np_mat):
        if np_mat is not None:
            np.save(file_name, np_mat)
            print('save npy file:{},size:{}'.format(file_name, np_mat.shape))
        else:
            print('empty! save nothing!')

    def detect(self):
        xmin, ymin, xmax, ymax = self.area
        frame_no = 0
        avg_fps = 0.0
        ret, ori_im = self.vdo.read()
        while ret:
            frame_no += 1
            start = time.time()
            im = ori_im[ymin:ymax, xmin:xmax]
            t1 = time.time()
            results = cascade_detector.detect(im, self.person_id)
            if self.write_det_txt:
                bbox_xyxyc = results
                if len(results) > 0:
                    bbox_xyxyc = bbox_xyxyc[bbox_xyxyc[:, 4] > 0.05]
                    for i, d in enumerate(bbox_xyxyc):
                        with open(self.det_txt, 'a') as f:
                            msg = '%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f\n' % (
                                frame_no, -1, d[0], d[1], d[2] - d[0], d[3] - d[1], d[4])
                            f.write(msg)
            t2 = time.time()
            if len(results) > 0:
                bbox_xywhcs = bbox_to_xywh_cls_conf(results, conf_thresh=0.05)
            else:
                bbox_xywhcs = []

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
                self.kc_tracker.update(frame_no, bbox_xywhcs, im)
            end = time.time()
            fps = 1 / (end - start)
            avg_fps += fps
            if frame_no % 100 == 0:
                print("cascade cost time: {}s, fps: {}, frame_no : {} track cost:{}".format(end - start, fps, frame_no,end - t2))

            if self.write_video:
                self.output.write(ori_im)
            if self.write_img:
                cv2.imwrite(os.path.join(self.img_dir, '{:06d}.jpg'.format(frame_no)), ori_im)

            ret, ori_im = self.vdo.read()

        self.vdo.release()
        if self.save_feature:
            self.saveFeature(self.features_npy, self.all_features)


if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    config_py = './Cascade/config/xz_cascade_mask_rcnn_x101_64x4d_fpn_0928.py'
    checkpoint_file = './Cascade/models/cascade_epoch_3.pth'
    cascade_detector = CascadeDetector(config_py, checkpoint_file)

    # path_dir = '/home/kcadmin/user/Dataset/level2_vedio'
    path_dir = './videos'
    seq = ['b2.mp4', 'b4.mp4']
    video_list = os.listdir(path_dir)
    video_file_list = [os.path.join(path_dir, vl) for vl in seq]
    video_file_list.sort(reverse=False)
    # video_file_list = ['/home/kcadmin/user/Dataset/level2_vedio/b2.mp4','/home/kcadmin/user/Dataset/level2_vedio/b4.mp4']

    print(video_file_list)
    start_time = datetime.now()
    print('start time:', start_time)
    for filename in video_file_list: #
        det = Detector(filename, max_cosine_distance=0.2,max_iou_distance=0.7, max_age=30, out_dir='results/res_20191001')
        det.save_feature = False
        det.write_det_txt = False
        det.use_tracker = True
        det.write_video = False
        det.write_bk = False
        print('################### start :', filename)
        det.open(filename)
        det.detect()
        det.kc_tracker.saveResult(det.mot_txt_filter)
        end_time = datetime.now()
        print('################### finish :', filename, end_time)
        print(' cost hour:', (end_time - start_time) / 3600)

