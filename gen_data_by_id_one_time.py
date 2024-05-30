from xml.dom import IndexSizeErr
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import h5py

from torch.utils.data import Dataset

import math
import argparse
import torch
import random
import cv2
import os, fnmatch
from models.origin_resnet import get_resnet_ms
from PIL import Image
from torchvision import transforms
from sklearn.metrics import mean_squared_error

rmse = lambda y_true, y_pred: math.sqrt(mean_squared_error(y_true, y_pred))

class FigureDataset(Dataset):
    def __init__(self, path: str) -> None:
        self.prefix = path
        self.files = fnmatch.filter(os.listdir(path), '*.jpg')
        self.files.sort()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_path = os.path.join(self.prefix, self.files[index])
        file_msg = os.path.splitext(self.files[index])[0]
        time, target = file_msg.split('-')
        target = int(target)
        img = Image.open(file_path)
        transformer = transforms.Compose([transforms.ToTensor()]) 
        img = transformer(img).unsqueeze(0)
        return img, target, time

def my_smooth(src):
    tmp = []
    tmp.append(src[0])
    for idx in range(1, len(src) - 1):
        tmp.append((tmp[idx - 1] + src[idx] + src[idx + 1]) / 3)
    tmp.append(src[len(src) - 1])
    return tmp

def main():
    num_classes = 1

    # init MSNet
    ms_pretrain_model = "checkpoint/2022_08_10_02_27_44cross-entropy/ckpt_9.757252683257262.pth"
    ms_weights_dict = torch.load(ms_pretrain_model, map_location="cpu")
    ms_state_dict = {}

    for k, v in ms_weights_dict['net'].items():
        if "module" in k:
            ms_state_dict[k[7:]] = v
        else:
            ms_state_dict[k] = v

    MSNet_model = get_resnet_ms("resnet32", num_classes)
    MSNet_model.load_state_dict(ms_state_dict, strict=True)
    MSNet_model.eval()
    print("MSNet successful using pretrain-weights.")

    # init OriginResNet
    origin_pretrain_model = "checkpoint/2022_08_10_02_25_55cross-entropy/ckpt_10.46196023958722.pth"
    origin_weights_dict = torch.load(origin_pretrain_model, map_location="cpu")
    origin_state_dict = {}

    for k, v in origin_weights_dict['net'].items():
        if "module" in k:
            origin_state_dict[k[7:]] = v
        else:
            origin_state_dict[k] = v

    OriginResNet_model = get_resnet_ms("OriginRes34", num_classes)
    OriginResNet_model.load_state_dict(origin_state_dict, strict=True)
    OriginResNet_model.eval()
    print("OriginResNet successful using pretrain-weights.")

    root = 'selected_dataset'
    
    items = os.listdir(root)

    for item in items:
        sub_path = os.path.join(root, item)
        if os.path.isdir(sub_path):
            ds = FigureDataset(sub_path)
            record_file_name = item + '.txt'

            record_file_path = os.path.join(sub_path, record_file_name)
            fw = open(record_file_path, 'w')

            time_list = []
            pred_ms_list = []
            pred_origin_list = []
            target_list = []

            for idx in range(len(ds)):
                img, target, time= ds[idx]
                time_list.append(time)
                target_list.append(target)

                # run MSNet model
                pred_ms = MSNet_model(img, 0)
                pred_ms = torch.squeeze(pred_ms).data.cpu().numpy()
                pred_ms_list.append(pred_ms)

                # run OriginResNet model
                pred_origin = OriginResNet_model(img, 0)
                pred_origin = torch.squeeze(pred_origin).data.cpu().numpy()
                pred_origin_list.append(pred_origin)
            
            rmse_origin = rmse(pred_origin_list, target_list)
            fw.write('origin rmse: {}\n'.format(rmse_origin))
            print('origin rmse: %f', rmse_origin)

            rmse_ms = rmse(pred_ms_list, target_list)
            fw.write('ms rmse: {}\n'.format(rmse_ms))
            print('ms rmse: %f', rmse_ms)

            pred_smooth_list = my_smooth(pred_ms_list)
            rmse_smooth = rmse(pred_smooth_list, target_list)
            fw.write('smooth rmse: {}\n'.format(rmse_smooth))
            print('smooth rmse: %f', rmse_smooth)

            for idx in range(len(time_list)):
                fw.write('{}\t{}\t{}\t{}\t{}\n'.format(time_list[idx], pred_ms_list[idx], pred_origin_list[idx], pred_smooth_list[idx], target_list[idx]))

            fw.close()

if __name__ == '__main__':
    main()
