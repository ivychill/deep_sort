import os,time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from .data.transforms import build_transforms
from .utils import set_random_seed, check_isfile, load_pretrained_weights
from . import models

torch.backends.cudnn.benchmark=True

## todo:
# 将bbox_tlwh, ori_img转化为numpy矩阵，批量处理，提取特征

class Extractor():

    def __init__(self, model_name, load_path, gpu_ids='0', use_gpu=True, height=256, width=128, seed=1):
        self.model_name = model_name
        self.load_path = load_path
        self.gpu_ids = gpu_ids
        self.use_gpu = use_gpu
        self.height = height
        self.width = width
        self.seed = seed

        self.model = models.build_model(
            name=self.model_name,
            num_classes=100,  # don't care this parameter when testing
        )
        self.model.eval()
        if check_isfile(self.load_path):
            load_pretrained_weights(self.model, self.load_path)
        if self.use_gpu:
            self.model = nn.DataParallel(self.model).cuda()

        _, self.transform_te = build_transforms(self.height, self.width,transforms=None)

    def imgListToTensor(self,img_list):
        imgs = []
        for inp in img_list:
            inp_ = Image.fromarray(np.uint8(inp)).convert('RGB')
            img = self.transform_te(inp_).unsqueeze(0)  # 将一张图转化为tensor
            imgs.append(img)
        # imgs_tensor = torch.cat(imgs, 0)
        return imgs

    def __call__(self, input,batch_size=10):
        '''
        :param input: detected images, numpy array
        :return: image features extracted from input
        '''

        set_random_seed(self.seed)
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_ids
        # 1. build test transform

        # numpy array to Image
        if isinstance(input,list):
            if len(input)==0:
                return np.array([])
            features = []
            batch_ = len(input)//batch_size
            batch_last = len(input)%batch_size
            if batch_last>0:
                batch_ +=1
            for ind in range(batch_):
                if ind==batch_-1 and batch_last>0: # 处理最后一个batch
                    input_batch = input[batch_size*ind:]
                    input_batch = input_batch+[input[0] for i in range(batch_size-batch_last)]
                    imgs_ = self.imgListToTensor(input_batch)
                    imgs = torch.cat(imgs_, 0)
                    if self.use_gpu:
                        img = imgs.cuda()
                    # 3. extract feature using model
                    extracted_feature = self.model(img)
                    features.append(extracted_feature[0:batch_last].detach().cpu().numpy())
                else:
                    input_batch = input[batch_size*ind:batch_size*(ind+1)]
                    imgs_ = self.imgListToTensor(input_batch)
                    imgs = torch.cat(imgs_, 0)
                    if self.use_gpu:
                        imgs = imgs.cuda()
                    # 3. extract feature using model
                    extracted_feature = self.model(imgs)
                    features.append(extracted_feature.detach().cpu().numpy())
            features = np.vstack(features)
            # imgs = torch.cat(imgs, 0)
            # st = time.time()
            # imgs = []
            # for inp in input:
            #     inp_ = Image.fromarray(np.uint8(inp)).convert('RGB')
            #     img = self.transform_te(inp_).unsqueeze(0)  # 将一张图转化为tensor
            #     imgs.append(img)
            # print('cat0 cost:', time.time() - st)
            # imgs = torch.cat(imgs,0)
            # print(imgs.shape)
            # print('cat1 cost:',time.time()-st)
        else:
            input_ = Image.fromarray(np.uint8(input)).convert('RGB')
            imgs = self.transform_te(input_).unsqueeze(0)  # 将一张图转化为tensor
            # 2. build model
            if self.use_gpu:
                imgs = imgs.cuda()
            # 3. extract feature using model
            extracted_feature = self.model(imgs)
            features = extracted_feature.detach().cpu().numpy()

        return features

if __name__ == "__main__":
    test_img_numpy = np.asarray(Image.open('1.jpg'), dtype='uint8')

    imgs = [test_img_numpy for i in range(5)]
    print(len(imgs))
    # etreactor = Extractor(model_name='osnet_x1_0',
    #                    load_path='/home/kcadmin/user/xz/deep-person-reid/checkpoint/model.pth.tar-460',
    #                    gpu_ids='0, 1')
    #
    # feature = etreactor(test_img_numpy)
    # print(feature.shape)
    # print(type(feature))
    # print(feature)