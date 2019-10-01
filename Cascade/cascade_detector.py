import mmcv
import os,time
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result
from mmdet.apis import init_detector, inference_detector

class CascadeDetector():
    def __init__(self,config_py,checkpoint,device='cuda:0'):
        # self.cfg = mmcv.Config.fromfile('configs/cascade_rcnn_x101_64x4d_fpn_1x.py')
        # self.cfg = mmcv.Config.fromfile(config_py)
        # self.cfg.model.pretrained = None
        # self.checkpoint_file = checkpoint
        # self.model = build_detector(self.cfg.model, test_cfg=self.cfg.test_cfg)
        # load_checkpoint(self.model, self.checkpoint_file)
        # self.device = device

        config_file = config_py
        checkpoint_file = checkpoint
        # build the model from a config file and a checkpoint file
        self.model = init_detector(config_file, checkpoint_file, device=device)

        # test a single image and show the results
        # img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once

    def detect(self,img,targetid=1):
        # input: ndarray img,int targetid
        # output: ndarray tlbrc (top left,botoom,right,confidence)
        result = inference_detector(self.model, img) # tlbrc list 80 class
        result = result[targetid-1] # person's position in the list
        return result




if __name__ == "__main__":
    config_py = '/home/kcadmin/user/xz/mmlab/mmdetction0.4/mmdetection/configs/cascade_rcnn_x101_64x4d_fpn_1x.py'
    checkpoint_file = '/home/kcadmin/user/xz/mmlab/mmdetction0.4/mmdetection/checkpoints/cascade_rcnn_x101_64x4d_fpn_1x_20181218-e2dc376a.pth'
    casdetector = CascadeDetector(config_py,checkpoint_file)
    img = mmcv.imread('/home/kcadmin/user/xz/mmlab/mmdetction0.4/mmdetection/result2/11195.jpg')
    res = casdetector.detect(img)
    aa = 1