import os
import cv2
import numpy as np
from datetime import datetime

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


class Detector(object):
    def __init__(self, opt, min_confidence=0.4, max_cosine_distance=0.2, max_iou_distance=0.7, max_age=30,
                 out_dir='res/'):
        self.vdo = cv2.VideoCapture()
        self.out_dir = out_dir
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            # os.makedirs(out_dir + '/imgs', exist_ok=True)

        # centerNet detector
        self.detector = detector_factory[opt.task](opt)
        # self.crop_detector = detector_factory[opt.task](crop_opt)

        self.kc_tracker = KCTracker(confidence_l=0.2, confidence_h=0.4,use_filter=True, max_cosine_distance=max_cosine_distance,
                                    max_iou_distance=max_iou_distance, max_age=max_age)

        _, filename = os.path.split(opt.vid_path)
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

    def open(self, video_path):
        if opt.input_type == 'webcam':
            self.vdo.open(opt.webcam_ind)
        elif opt.input_type == 'ipcam':
            # load cam key, secret
            with open("cam_secret.txt") as f:
                lines = f.readlines()
                key = lines[0].strip()
                secret = lines[1].strip()
            self.vdo.open(opt.ipcam_url.format(key, secret, opt.ipcam_no))
        else:  # video
            assert os.path.isfile(opt.vid_path), "Error: path error"
            self.vdo.open(opt.vid_path)

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
        # crop_top, crop_left, crop_h, crop_w = 0, 650, 150, 520
        ret, ori_im = self.vdo.read()
        while ret:
            frame_no += 1
            start = time.time()
            im = ori_im[ymin:ymax, xmin:xmax]
            t1 = time.time()
            results_big = self.detector.run(im)['results'][self.person_id]
            results = results_big
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
                print("detect cost time: {}s, fps: {}, frame_no : {} track cost:{}".format(end - start, fps, frame_no,
                                                                                           end - t2))

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
                       max_iou_distance=0.7, max_age=30, out_dir='results/res_20191001')
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
