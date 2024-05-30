from fileinput import filename
from pathlib import Path

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="resnet32",
        #default="OriginRes34",
        choices=[
            "resnet32",
            "resnet50",
            "resnet101",
            "OriginRes50",
            "OriginRes34"
        ],
        help="Model to use",
    )

    parser.add_argument('--weights', default="checkpoint/2022_08_10_02_27_44cross-entropy/ckpt_9.757252683257262.pth", type=str, help='initial weights path')

    parser.add_argument('--smooth', action="store_true", default=False)

    args = parser.parse_args()

    return args

def my_smooth(src):
    for idx in range(1, len(src) - 1):
        src[idx] = (src[idx - 1] + src[idx] + src[idx + 1]) / 3
    return src


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    args = parse_args()
    num_classes = 1
    model = get_resnet_ms(args.model, num_classes)
    weights_dict = torch.load(args.weights, map_location="cpu")
    new_state_dict = {}

    for k, v in weights_dict['net'].items():
        if "module" in k:
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=True)
    # model = model.to(device)
    model.eval()
    print("Successful using pretrain-weights.")

    root = 'selected_dataset'
    
    items = os.listdir(root)

    for item in items:
        sub_path = os.path.join(root, item)
        if os.path.isdir(sub_path):
            ds = FigureDataset(sub_path)
            record_file_name = args.model
            if args.smooth:
                record_file_name += '_smooth'
            record_file_name += '_val.txt'

            record_file_path = os.path.join(sub_path, record_file_name)
            fo = open(record_file_path, 'w')

            pred_list = []
            target_list = []

            for idx in range(len(ds)):
                img, target, time= ds[idx]
                pred = model(img, 0)
                pred = torch.squeeze(pred).data.cpu().numpy()
                pred_list.append(pred)
                target_list.append(target)
                fo.write('{}, {}, {}\n'.format(time, pred, target))
            
            cal_rmse = rmse(pred_list, target_list)
            fo.write('rmse: {}\n'.format(cal_rmse))
            print('rmse: %f', cal_rmse)
            fo.close()
            pass
        pass
    pass

if __name__ == '__main__':
    main()
