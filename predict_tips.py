# -*- coding: utf-8 -*-

import torch
from torchvision import transforms
import os
import cv2
import time
import numpy as np
from pathlib import Path

from pse import decode as pse_decode

class Pytorch_model:
    def __init__(self, model_path, net, scale, gpu_id=None):
        '''
        初始化pytorch模型
        :param model_path: 模型地址(可以是模型的参数或者参数和计算图一起保存的文件)
        :param net: 网络计算图，如果在model_path中指定的是参数的保存路径，则需要给出网络的计算图
        :param img_channel: 图像的通道数: 1,3
        :param gpu_id: 在哪一块gpu上运行
        '''
        self.scale = scale
        if gpu_id is not None and isinstance(gpu_id, int) and torch.cuda.is_available():
            self.device = torch.device("cuda:{}".format(gpu_id))
        else:
            self.device = torch.device("cpu")
        #self.net = torch.load(model_path, map_location=self.device)['state_dict']
        net.load_state_dict(torch.load(model_path))
        self.net = net
        print('device:', self.device)

        if net is not None:
            # 如果网络计算图和参数是分开保存的，就执行参数加载
            net = net.to(self.device)
            net.scale = scale
            try:
                sk = {}
                for k in self.net:
                    sk[k[7:]] = self.net[k]
                net.load_state_dict(sk)
            except:
                net.load_state_dict(self.net)
            self.net = net
            print('load models')
        self.net.eval()

    def predict(self, img: str, long_size: int = 2240):
        '''
        对传入的图像进行预测，支持图像地址,opecv 读取图片，偏慢
        :param img: 图像地址
        :param is_numpy:
        :return:
        '''
        assert os.path.exists(img), 'file is not exists'
        img = cv2.imread(img)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        scale = long_size / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale)
        # 将图片由(w,h)变为(1,img_channel,h,w)
        tensor = transforms.ToTensor()(img)
        tensor = tensor.unsqueeze_(0)

        tensor = tensor.to(self.device)
        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.time()
            preds = self.net(tensor)
            preds, boxes_list = pse_decode(preds[0], self.scale)
            scale = (preds.shape[1] / w, preds.shape[0] / h)
            # print(scale)
            # preds, boxes_list = decode(preds,num_pred=-1)
            if len(boxes_list):
                boxes_list = boxes_list / scale
            torch.cuda.synchronize()
            t = time.time() - start
        return preds, boxes_list, t

class Pytorch_model2:
    def __init__(self, model_path, net, scale, gpu_id=None):
        '''
        初始化pytorch模型
        :param model_path: 模型地址(可以是模型的参数或者参数和计算图一起保存的文件)
        :param net: 网络计算图，如果在model_path中指定的是参数的保存路径，则需要给出网络的计算图
        :param img_channel: 图像的通道数: 1,3
        :param gpu_id: 在哪一块gpu上运行
        '''
        self.scale = scale
        if gpu_id is not None and isinstance(gpu_id, int) and torch.cuda.is_available():
            self.device = torch.device("cuda:{}".format(gpu_id))
        else:
            self.device = torch.device("cpu")
        net.load_state_dict(torch.load(model_path))
        self.net = net
        self.net.to(self.device)
        print('device:', self.device)
        self.net.eval()

    def predict(self, img: str, long_size: int = 2240):
        '''
        对传入的图像进行预测，支持图像地址,opecv 读取图片，偏慢
        :param img: 图像地址
        :param is_numpy:
        :return:
        '''
        assert os.path.exists(img), 'file is not exists'
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        scale = long_size / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale)
        # 将图片由(w,h)变为(1,img_channel,h,w)
        tensor = transforms.ToTensor()(img)
        tensor = tensor.unsqueeze_(0)

        tensor = tensor.to(self.device)
        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.time()
            preds = self.net(tensor)
            preds, boxes_list = pse_decode(preds[0], self.scale)
            scale = (preds.shape[1] / w, preds.shape[0] / h)
            # print(scale)
            # preds, boxes_list = decode(preds,num_pred=-1)
            if len(boxes_list):
                boxes_list = boxes_list / scale
            #torch.cuda.synchronize()
            t = time.time() - start
        return preds, boxes_list, t


def _get_annotation(label_path):
    boxes = []
    with open(label_path, encoding='utf-8', mode='r') as f:
        for line in f.readlines():
            params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')
            try:
                label = params[8]
                if label == '*' or label == '###':
                    continue
                x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, params[:8]))
                boxes.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            except:
                print('load label failed on {}'.format(label_path))
    return np.array(boxes, dtype=np.float32)


if __name__ == '__main__':
    from configs import config_tips as config
    from models import PSENet
    from utils.utils import show_img, draw_bbox

    #model_path = 'outp ut/psenet_icd2015_resnet152_author_crop_adam_warm_up_myloss/best_r0.714011_p0.708214_f10.711100.pth'
    #model_path = 'output/psenet_icd2015_new_loss/final.pth'
    model_path = 'checkpoint/pse_epoch_30.pth'
    #image_files = Path("/home/peizhao/data/icdar/2019/tips/test/img").rglob('*.jpg')
    #image_files = Path("/home/peizhao/data/temp/doc").rglob("*.jpg")
    image_files = Path("/home/peizhao/data/temp").rglob("*.png")
    # image_files = Path("/home/peizhao/data/temp").rglob("*.jpg")
    #image_files = Path("/home/peizhao/data/temp/test").rglob("*.jpg")
    # 初始化网络
    #net = PSENet(backbone='resnet152', pretrained=False, result_num=config.n)
    net = PSENet(backbone='resnet50', pretrained=False, result_num=3)
    model = Pytorch_model2(model_path, net=net, scale=1, gpu_id=0)

    for item in image_files:
        preds, boxes_list, t = model.predict(str(item))
        img = draw_bbox(str(item), boxes_list, color=(0, 0, 255))
        h, w = img.shape[:2]
        scale = 640 / max(h, w)
        img_size = cv2.resize(img, None, fx=scale, fy=scale)
        cv2.imshow("result",img_size)
        cv2.waitKey(0)
