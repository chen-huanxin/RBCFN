# from more_itertools import sort_together
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import h5py

from torch.utils.data import Dataset

from torchvision import transforms

import random
import cv2
import os
from PIL import Image


# X = ["r", "s", "t", "u", "v", "w", "x", "y", "z"]
# Y = [ 2017041606, 2015010101, 2016010101, 2019010101, 2022020202, 2010010101, 2012121212, 2000010203, 2011020202]
# s = sort_together([Y, X])[1]
# print(list(s))

class TCIRDataSet(Dataset): # 继承pytorch的Dataset类
    def __init__(self, dataset_dir: str) -> None:
        self.dataset_dir = Path(dataset_dir)

        # There are 2 keys in the HDF5:
        # matrix: N x 201 x 201 x 4 HDF5 dataset. One can load this with python numpy.
        # info: HDF5 group. One can load this with python package pandas.
        # 用pandas读取进来，就会是非常容易操作的表格形式


        # load "info" as pandas dataframe
        self.data_info = pd.read_hdf(self.dataset_dir, key="info", mode='r')
        # self.data_info.to_excel('output-CPAC.xls')        

        # load "matrix" as numpy ndarray, this could take longer times
        # with h5py.File(self.dataset_dir, 'r') as hf:
        #     self.data_matrix = hf['matrix'][:]
        # 修改了h5py文件的加载方式，看看能不能加载大文件
        self.data_matrix = h5py.File(self.dataset_dir, 'r')['matrix']

    def __len__(self):
        return len(self.data_info)

    def AvoidDamagedVal(self, matrix):
        NanVal = np.where(matrix==np.NaN)
        LargeVal = np.where(matrix>1000)
        DemagedVal = [NanVal, LargeVal]
        for item in DemagedVal:
            for idx in range(len(item[0])):
                i = item[0][idx]
                j = item[1][idx]
                allValidPixel = []
                for u in range(-2,2):
                    for v in range(-2,2):
                        if (i+u) < 201 and (j+v) < 201 and not np.isnan(matrix[i+u,j+v]) and not matrix[i+u,j+v] > 1000:
                            allValidPixel.append(matrix[i+u,j+v])
                if len(allValidPixel) != 0:
                    matrix[i][j] = np.mean(allValidPixel)

        return matrix

    def __getitem__(self, index):        
        id = self.data_info.iloc[index].loc['ID']
        vmax = self.data_info.iloc[index].loc['Vmax']
        Lon = self.data_info.iloc[index].loc['lon']
        Lat = self.data_info.iloc[index].loc['lat']
        Time = self.data_info.iloc[index].loc['time']

        # Slice1: IR
        # Slice2: Water vapor
        # Slice3: VIS
        # Slice4: PMW

        ch_slice = self.data_matrix[index][:, :, 0]
        ch_slice1 = self.data_matrix[index][:, :, 1]
        ch_slice3 = self.data_matrix[index][:, :, 3]

        img = np.zeros((201, 201, 3))
        ch_slice = self.AvoidDamagedVal(ch_slice)
        ch_slice1 = self.AvoidDamagedVal(ch_slice1)
        ch_slice3 = self.AvoidDamagedVal(ch_slice3)

        img[:, :, 0] = ch_slice # IR
        img[:, :, 1] = ch_slice1# Water vapor
        img[:, :, 2] = ch_slice3# PMW

        img = img.astype(np.uint8)
        return img, vmax, Lon, Lat, Time, id

    # def __del__(self):
    #     self.data_matrix.close()

class MySubset_WS(Dataset):
    """ 划分数据集之后重新设置预处理操作 """

    def __init__(self, dataset: Dataset, transform=None) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def _is_leap_year(self, year):
        if year % 4 != 0 or (year % 100 == 0 and year % 400 != 0):
            return False
        return True

    def get_scaled_date_ratio(self, year, month, day):
        r'''
        scale date to [-1,1]
        '''
        days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        total_days = 365
        # year = date_time.year
        # month = date_time.month
        # day = date_time.day
        if self._is_leap_year(year):
            days[1] += 1
            total_days += 1

        assert day <= days[month - 1]
        sum_days = sum(days[:month - 1]) + day
        assert sum_days > 0 and sum_days <= total_days

        # Transform to [-1,1]
        return (sum_days / total_days) * 2 - 1

    def GetItemMultiScale(self, Lon, lat):


        return 1

    def __getitem__(self, index):
        img, label, Lon, Lat, time, id = self.dataset[index]

        if self.transform is not None:
            img = self.transform(img)

        label = int(label)

        return img, label, time, id

def cmp_time(item1, item2):
    if item1[2] < item2[2]:
        return -1
    elif item1[2] > item2[2]:
        return 1
    else:
        return 0

def cmp_id(item1, item2):
    if item1[1] < item2[1]:
        return -1
    elif item1[1] > item2[1]:
        return 1
    else:
        return 0

def save_img(img):
    # save image
    print(img.shape)
    x_local = img[48:(48 + 128), 48:(48 + 128),:]
    print('x_local type: ')
    print(type(x_local))    # class 'torch.Tensor'
    print(x_local.shape)   # [1,3,128,128]
    cv2.imwrite('pic1.jpg', x_local)

    # Centre Crop local patch 64 x 64
    x_local2 = img[ 80:(80 + 64), 80:(80 + 64), :]
    print('x_local2 type: ')
    print(type(x_local2))   # class 'torch.Tensor'
    print(x_local2.shape)  # [1,3,64,64]
    cv2.imwrite('pic2.jpg', x_local2)

def sort_data(dataset, mylist):
    for idx in range(len(dataset)):
        img, label, time, id = dataset[idx]
        # sort
        mylist.append([label, id, int(time)])

    sorted(mylist, key=lambda x: x[2])
    sorted(mylist, key=lambda x: x[1])

    print(mylist)

def select_id(dataset, num: int):
    myset = set()
    for idx in range(len(dataset)):
        _, _, _, id = dataset[idx]
        # sort
        myset.add(id)

    id_list = list(myset)
    rst = random.sample(id_list, num)
    return set(rst)

def save_my_dataset(dataset, id_set, root):
    for item in id_set:
        dir_path = os.path.join(root, item)
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

    for idx in range(len(dataset)):
        img, label, time, id = dataset[idx]
        if id in id_set:
            save_name = time + '-' + str(label) + '.jpg'
            save_path = os.path.join(root, id)
            # cv2.imwrite(os.path.join(save_path, save_name), img)
            img.save(os.path.join(save_path, save_name))

def main():
    data_path = '/home/chenhuanxin/datasets/TCIR-SPLT/TCIR-test.h5'

    transform_test = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(224),
            ]
        )
    ds = TCIRDataSet(data_path)
    test_set = MySubset_WS(ds, transform=transform_test)
    print(len(test_set))
    selected_id = select_id(test_set, 20)
    root = 'selected_dataset'
    save_my_dataset(test_set, selected_id, root)

if __name__ == '__main__':
    main()
