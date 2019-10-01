from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
import numpy as np
import torch
import json
import os

import torch.utils.data as data
# kc/
#    annotations/xx.json
#    images/xx.jpg
# cd ../src
# # train
# python main.py ctdet --exp_id kc_dla_512 --dataset kc --input_res 512 --num_epochs 70 --lr_step 45,60 --gpus 2,3 --save_all
# # test
# python test.py ctdet --exp_id kc_dla_512 --dataset kc --input_res 512 --resume --gpus 3
# # flip test
# python test.py ctdet --exp_id kc_dla_512 --dataset kc --input_res 512 --resume --flip_test --gpus 3
# cd ..

class Kc_annotate(data.Dataset):
  num_classes = 1
  default_resolution = [384, 384]
  mean = np.array([0.485, 0.456, 0.406],
                   dtype=np.float32).reshape(1, 1, 3)
  std  = np.array([0.229, 0.224, 0.225],
                   dtype=np.float32).reshape(1, 1, 3)
  
  def __init__(self, opt, split):
    super(Kc_annotate, self).__init__()
    self.data_dir = os.path.join(opt.data_dir, 'kc')
    self.img_dir = os.path.join(self.data_dir, 'images')
    _ann_name = {'train': 'train', 'val': 'test'}
    self.annot_path = os.path.join(
      self.data_dir, 'annotations', 
      'kc_{}.json').format(_ann_name[split])
    self.max_objs = 50
    self.class_name = ['__background__', 'pedestrian']
    self._valid_ids = np.arange(1, 5, dtype=np.int32)
    self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
    self._data_rng = np.random.RandomState(123)
    self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
    self._eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
    self.split = split
    self.opt = opt

    print('==> initializing kc annotate {} data.'.format(_ann_name[split]))
    self.coco = coco.COCO(self.annot_path)
    self.images = sorted(self.coco.getImgIds())
    self.num_samples = len(self.images)

    print('Loaded {} {} samples'.format(split, self.num_samples))

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def convert_eval_format(self, all_bboxes):
    detections = [[[] for __ in range(self.num_samples)] \
                  for _ in range(self.num_classes + 1)]
    for i in range(self.num_samples):
      img_id = self.images[i]
      for j in range(1, self.num_classes + 1):
        if isinstance(all_bboxes[img_id][j], np.ndarray):
          detections[j][i] = all_bboxes[img_id][j].tolist()
        else:
          detections[j][i] = all_bboxes[img_id][j]
    return detections

  def __len__(self):
    return self.num_samples

  def save_results(self, results, save_dir):
    json.dump(self.convert_eval_format(results), 
              open('{}/results.json'.format(save_dir), 'w'))

  def get_bbox(self, bbox_wh):
    return [bbox_wh[0], bbox_wh[1], bbox_wh[0]+bbox_wh[2], bbox_wh[1]+bbox_wh[3]]

  def voc_ap(self, rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
      # 11 point metric
      ap = 0.
      for t in np.arange(0., 1.1, 0.1):
        if np.sum(rec >= t) == 0:
          p = 0
        else:
          p = np.max(prec[rec >= t])
          # print(t, p)
        ap = ap + p / 11.
    else:
      # correct AP calculation
      # first append sentinel values at the end
      mrec = np.concatenate(([0.], rec, [1.]))
      mpre = np.concatenate(([0.], prec, [0.]))

      # compute the precision envelope
      for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

      # to calculate area under PR curve, look for points
      # where X axis (recall) changes value
      i = np.where(mrec[1:] != mrec[:-1])[0]

      # and sum (\Delta recall) * prec
      ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

  def run_eval(self, results, save_dir):
    # result_json = os.path.join(save_dir, "results.json")
    # detections  = self.convert_eval_format(results)
    # json.dump(detections, open(result_json, "w"))
    self.save_results(results, save_dir)
    os.system('python tools/reval.py ' + \
              '{}/results.json --imdb kc_test'.format(save_dir))

    return

    ovthresh = 0.5

    # read detected bbox
    format_res_dict = {cls_idx: [] for cls_idx in range(1, self.num_classes)}
    for img_id, res_dict in results.items():
      for cls_idx in range(1, self.num_classes):
        if len(res_dict[cls_idx]) > 0:
          for lt_x, lt_y, rb_x, rb_y, score in res_dict[cls_idx]:
            format_res_dict[cls_idx].append([img_id, score, lt_x, lt_y, rb_x, rb_y])

    # print(format_res_dict)

    # read ground-truth bbox
    format_gt_dict = {cls_idx: {img_id: {'bbox': [], 'det': []}  for img_id in self.images} for cls_idx in range(1, self.num_classes)}
    for img_id in self.images:
      anno_ids = self.coco.getAnnIds(img_id)
      annos = self.coco.loadAnns(anno_ids)
      for anno in annos:
        format_gt_dict[anno['category_id']][img_id]['bbox'].append(self.get_bbox(anno['bbox']))
        format_gt_dict[anno['category_id']][img_id]['det'].append(False)

    # print(format_gt_dict)

    for cls_idx in range(1, self.num_classes):

      npos = sum([len(bboxes) for img_id in self.images for bboxes in format_gt_dict[cls_idx][img_id]['bbox']])

      image_ids = [x[0] for x in format_res_dict[cls_idx]]
      confidence = np.array([float(x[1]) for x in format_res_dict[cls_idx]])
      BB = np.array([[float(z) for z in x[2:]] for x in format_res_dict[cls_idx]])

      nd = len(image_ids)
      tp = np.zeros(nd)
      fp = np.zeros(nd)

      if BB.shape[0] > 0:
        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        for d in range(nd):
          R = format_gt_dict[cls_idx][image_ids[d]]
          bb = BB[d, :].astype(float)
          ovmax = -np.inf
          BBGT = np.array(R['bbox']).astype(float)

          if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

          if ovmax > ovthresh:
            if not R['det'][jmax]:
              tp[d] = 1.
              R['det'][jmax] = 1
            else:
              fp[d] = 1.
          else:
            fp[d] = 1.

      # compute precision recall
      fp = np.cumsum(fp)
      tp = np.cumsum(tp)
      rec = tp / float(npos)
      # avoid divide by zero in case the first detection matches a difficult
      # ground truth
      prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
      ap = self.voc_ap(rec, prec, True)

      print(('AP for {} = {:.4f}'.format(self.class_name[cls_idx], ap)))