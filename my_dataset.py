import os.path
import math
import PIL.Image
import numpy as np
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset
from skimage.transform import warp_polar
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob
import pandas as pd
import json
from pylab import *
from collections import Counter
import pickle
import h5py
from mpl_toolkits.basemap import Basemap

class MyDataSetTCIR(Dataset):
    """ 自定义数据集 """

    def __init__(self, dataset_dir: str, multi_mode=False) -> None:

        self.dataset_dir = Path(dataset_dir)
        self.multi_mode = multi_mode
        # load "info" as pandas dataframe
        self.data_info = pd.read_hdf(self.dataset_dir, key="info", mode='r')

        # load "matrix" as numpy ndarray, this could take longer times
        # with h5py.File(self.dataset_dir, 'r') as hf:
        #     self.data_matrix = hf['matrix'][:]
        # 修改了h5py文件的加载方式，看看能不能加载大文件
        self.data_matrix = h5py.File(self.dataset_dir, 'r')['matrix']

        showPosition = 0
        if showPosition == 1:
            all_lon = []
            all_lat = []
            all_vmax = []
            all_time = []
            # Show all the location of the Typhoon
            #for i in range(39811):

            for i in range(7569):
                vmax = self.data_info.iloc[i].loc['Vmax']
                Lon = self.data_info.iloc[i].loc['lon']
                Lat = self.data_info.iloc[i].loc['lat']
                Time = self.data_info.iloc[i].loc['time']

                all_vmax.append(vmax)
                all_lon.append(Lon)
                all_lat.append(Lat)
                all_time.append(int(Time[:4]))

            all_unique_time = np.unique(all_time)

            # 创建一个地图用于绘制。我们使用的是墨卡托投影，并显示整个世界。
            #m = Basemap(projection='merc', llcrnrlat=-50, urcrnrlat=65, llcrnrlon=-165, urcrnrlon=155, lat_ts=20,
            #            resolution='c')

            m = Basemap(projection='robin', lat_0=0, lon_0=0, resolution='i',
                            area_thresh=5000.0)

            # 绘制海岸线，以及地图的边缘
            m.drawcoastlines()
            m.drawmapboundary()
            m.drawcountries()
            m.drawstates()
            m.drawcounties()

            # Convert coords to projected place in figur
            x, y = m(all_lon, all_lat)
            m.scatter(x, y, 1, marker='.', c=all_vmax, cmap=plt.cm.Set1)
            cb = m.colorbar()

            plt.show()

    def __len__(self):
        return len(self.data_info)

    def AvoidDamagedVal(self, matrix):
        # matrix_shape = matrix.shape

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
        # a = np.where(matrix==np.nan or matrix > 1000)
        # if not a[0]:
        #     return matrix

        # for i in range(matrix_shape[0]):
        #     for j in range(matrix_shape[1]):
        #         if np.isnan(matrix[i][j]) or matrix[i][j] > 1000:
        #             allValidPixel = []
        #             for u in range(-2,2):
        #                 for v in range(-2,2):
        #                     if (i+u) < 201 and (j+v) < 201 and not np.isnan(matrix[i+u,j+v]) and not matrix[i+u,j+v] > 1000:
        #                         allValidPixel.append(matrix[i+u,j+v])
        #             if len(allValidPixel) != 0:
        #                 matrix[i][j] = np.mean(allValidPixel)

        return matrix

    # def AvoidNan(self, matrix):
    #     matrix_shape = matrix.shape

    #     a = np.where(matrix==np.nan)
    #     if not a[0]:
    #         return matrix

    #     for i in range(matrix_shape[0]):
    #         for j in range(matrix_shape[1]):
    #             if np.isnan(matrix[i][j]):
    #                 allValidPixel = []
    #                 for u in range(-2,2):
    #                     for v in range(-2,2):
    #                         if (i+u) < 201 and (j+v) < 201 and not np.isnan(matrix[i+u,j+v]) and not matrix[i+u,j+v] > 1000:
    #                             allValidPixel.append(matrix[i+u,j+v])

    #                 matrix[i][j] = np.mean(allValidPixel)

    #     return matrix

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
        ch_slice2 = self.data_matrix[index][:, :, 2]
        ch_slice3 = self.data_matrix[index][:, :, 3]

        #plt.imshow(ch_slice3)
        #plt.show()

        # Consider The Great IR
        #ch_slice = self.AvoidNan(ch_slice)
        #ch_slice1 = self.AvoidNan(ch_slice1)
        #ch_slice2 = self.AvoidNan(ch_slice2)
        #ch_slice3 = self.AvoidNan(ch_slice3)

        #ch_slice = self.data_matrix[index][:, :, 0]
        #ch_slice = self.data_matrix[index][:, :, 0]

        img = np.zeros((201, 201, 3))
        if self.multi_mode:
            ch_slice = self.AvoidDamagedVal(ch_slice)
            ch_slice1 = self.AvoidDamagedVal(ch_slice1)
            #ch_slice2 = self.AvoidDamagedVal(ch_slice2)
            ch_slice3 = self.AvoidDamagedVal(ch_slice3)

            img[:, :, 0] = ch_slice # IR
            img[:, :, 1] = ch_slice1# Water vapor
            img[:, :, 2] = ch_slice3# PMW
        else: 
            img[:, :, 0] = ch_slice
            img[:, :, 1] = ch_slice
            img[:, :, 2] = ch_slice

        img = img.astype(np.uint8)
        img = PIL.Image.fromarray(img)
        return img, vmax, Lon, Lat, Time, id

    # def __del__(self):
    #     self.data_matrix.close()

class MySubset(Dataset):
    """ 划分数据集之后重新设置预处理操作 """

    def __init__(self, dataset: Dataset, transform=None) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label, Lon, Lat, Time = self.dataset[index]
        label = Transform_WindSpeed_Classes2(label)

        if self.transform is not None:
            img = self.transform(img)
        return img, label

class MySubsetGeo(Dataset):
    """ 划分数据集之后重新设置预处理操作 """

    def __init__(self, dataset: Dataset, transform=None) -> None:
        self.dataset = dataset
        self.transform = transform

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

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label, Lon, Lat, Time = self.dataset[index]

        # lon {-180, 180}, lat {90, -90}
        # Transform to [-1,1]
        Lon /= 180.0
        Lat /= 90.0

        Year = Time[:4]
        Month = Time[4:6]
        Day = Time[6:8]
        Hour = Time[8:]

        Day_embedding = self.get_scaled_date_ratio(int(Year), int(Month), int(Day))

        Hour_embedding = (int(Hour) / 24.0) * 2 - 1 # Transform to [-1,1]

        loc_time_embeding = torch.from_numpy(np.array([Lon, Lat, Day_embedding, Hour_embedding]))

        #a = torch.sin(math.pi * loc_time_embeding)
        #b = torch.cos(math.pi * loc_time_embeding)

        # Feature Encoding
        spatial_feats = torch.cat((torch.sin(math.pi * loc_time_embeding), torch.cos(math.pi * loc_time_embeding)))

        spatial_feats = spatial_feats.float()
        # feats_date = torch.cat((torch.sin(math.pi * date_ip.unsqueeze(-1)),
        #                         torch.cos(math.pi * date_ip.unsqueeze(-1))), concat_dim)

        label = Transform_WindSpeed_Classes2(label)

        if self.transform is not None:
            img = self.transform(img)

        return img, label, spatial_feats

class MySubsetGeo_WS(Dataset):
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
        img, label, Lon, Lat, Time = self.dataset[index]
        # lon {-180, 180}, lat {90, -90}
        # Transform to [-1,1]
        Lon /= 180.0
        Lat /= 90.0

        Year = Time[:4]
        Month = Time[4:6]
        Day = Time[6:8]
        Hour = Time[8:]

        Day_embedding = self.get_scaled_date_ratio(int(Year), int(Month), int(Day))

        Hour_embedding = (int(Hour) / 24.0) * 2 - 1  # Transform to [-1,1]

        Month_embedding = int(Month) / 12

        #loc_time_embeding = torch.from_numpy(np.array([Lon, Lat, Day_embedding, Hour_embedding]))
        #loc_time_embeding = torch.from_numpy(np.array([Lon, Lat, Day_embedding]))
        #loc_time_embeding = torch.from_numpy(np.array([Lon, Lat, Month_embedding]))
        #loc_time_embeding = torch.from_numpy(np.array([Lon, Lat, Month_embedding]))
        loc_time_embeding = torch.from_numpy(np.array([Lon, Lat]))

        # a = torch.sin(math.pi * loc_time_embeding)
        # b = torch.cos(math.pi * loc_time_embeding)

        # Feature Encoding
        spatial_feats = torch.cat((torch.sin(math.pi * loc_time_embeding), torch.cos(math.pi * loc_time_embeding)))
        # spatial_feats = torch.cat(Lon, Lat)
        spatial_feats = loc_time_embeding

        spatial_feats = spatial_feats.float()

        if self.transform is not None:
            img = self.transform(img)

        label = int(label)

        return img, label, spatial_feats

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
        img, label, Lon, Lat, Time, id = self.dataset[index]

        if self.transform is not None:
            img = self.transform(img)

        label = int(label)

        classification_label = Transform_WindSpeed_Classes2(label)

        return img, label, classification_label, Time, id

class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, TxTFile, transform=None):

        # Read From TxTFile
        images_path = []
        ALL_MSW = []
        ALL_TC_TYPE = []
        ALL_MSLP = []

        with open(TxTFile, "r") as f:
            for line in f:
                images_path_ = line
                images_path.append(images_path_.strip())
                #images_para_ = images_path_.replace("Y_TC_ESTIMATE/SelectPic", "Y_TC_ESTIMATE/ALL/Parameter/MSW")
                #images_para_ = images_path_.replace("Y_TC_ESTIMATE/SelectPic/IR_TC_Type", "Y_TC_ESTIMATE/ALL/Parameter/TC_Type")
                images_para_ = images_path_.replace("Y_TC_ESTIMATE_HUR/SelectPic/IR_MSW/","Y_TC_ESTIMATE_HUR/ALL/Parameter/MSW/")
                #images_para_ = images_path_.replace("Y_TC_ESTIMATE/SelectPic/IR_MSLP","Y_TC_ESTIMATE/ALL/Parameter/MSLP")

                images_para_ = images_para_.replace(".png\n", ".txt")
                with open(images_para_, "r") as f1:
                    TC_TYPE = f1.readline()
                    ALL_TC_TYPE.append(float(TC_TYPE))

        #b = np.unique(ALL_TC_TYPE)
        #hist, bins = np.histogram(ALL_TC_TYPE, bins=7, range=(0, 6))

        # self.RootImageDir = "/mnt/564C2A944C2A6F45/DataSet/Y_TC_ESTIMATE/SelectPic/IR_TC_Type"
        # self.RootParameterDir = "/mnt/564C2A944C2A6F45/DataSet/Y_TC_ESTIMATE/ALL/Parameter/TC_Type"
        #
        self.RootImageDir = "/mnt/564C2A944C2A6F45/DataSet/Y_TC_ESTIMATE/SelectPic/IR_MSW"
        self.RootParameterDir = "/mnt/564C2A944C2A6F45/DataSet/Y_TC_ESTIMATE/ALL/Parameter/MSW"

        #self.RootImageDir = "/mnt/564C2A944C2A6F45/DataSet/Y_TC_ESTIMATE/SelectPic/IR_MSLP"
        #self.RootParameterDir = "/mnt/564C2A944C2A6F45/DataSet/Y_TC_ESTIMATE/ALL/Parameter/MSLP"

        self.images_path = images_path
        self.images_class = ALL_TC_TYPE
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def Usual_GetItem(self, item):
        img = Image.open(self.images_path[item])

        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def FANet_GetItem(self, item):
        img = Image.open(self.images_path[item])

        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        # Perform Polar Transform: [3, 224, 224]
        Polar_Image = np.asarray(img)
        # [3, 224, 224] ---> [224, 224, 3]
        Polar_Image = np.transpose(Polar_Image, [1, 2, 0])
        # Polar Transform:
        Polar_Image = warp_polar(Polar_Image, radius=112, multichannel=True)
        # plt.imshow(Polar_Image)
        # plt.show()

        Polar_Image = cv2.resize(Polar_Image, (224, 224))
        # [224, 224, 3] ---> [3, 224, 224]
        Polar_Image = np.transpose(Polar_Image, [2, 1, 0])
        Polar_Image = torch.from_numpy(Polar_Image)

        All_Image = torch.cat((img, Polar_Image), 0)

        return All_Image, label

    def __getitem__(self, item):

        #return self.Usual_GetItem(item)
        return self.FANet_GetItem(item)

        # img = Image.open(self.images_path[item])
        #
        # # RGB为彩色图片，L为灰度图片
        # if img.mode != 'RGB':
        #     raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        # label = self.images_class[item]
        #
        # if self.transform is not None:
        #     img = self.transform(img)
        #
        # # Perform Polar Transform: [3, 224, 224]
        # Polar_Image = np.asarray(img)
        # # [3, 224, 224] ---> [224, 224, 3]
        # Polar_Image = np.transpose(Polar_Image,[1,2,0])
        # # Polar Transform:
        # Polar_Image = warp_polar(Polar_Image, radius=112, multichannel=True)
        # #plt.imshow(Polar_Image)
        # #plt.show()
        #
        # Polar_Image = cv2.resize(Polar_Image, (224, 224))
        # # [224, 224, 3] ---> [3, 224, 224]
        # Polar_Image = np.transpose(Polar_Image, [2,1,0])
        # Polar_Image = torch.from_numpy(Polar_Image)
        #
        # All_Image = torch.cat((img, Polar_Image),0)
        #
        # return All_Image, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels, dtype=torch.float)

        return images, labels

class MyDataSetTypeClassification(Dataset):
    """自定义数据集"""

    def __init__(self, TxTFile, transform=None):

        # Read From TxTFile
        images_path = []
        ALL_MSW = []
        ALL_TC_TYPE = []
        ALL_MSLP = []

        image_para_root = "/media/dell/564C2A944C2A6F45/DataSet/Y_TC_ESTIMATE_HUR/ALL/Parameter/Type"

        with open(TxTFile, "r") as f:
            for line in f:
                images_path_ = line

                imageName = images_path_.split("/")
                imageFileName = imageName[-1]
                imageFileYear = imageName[-2]
                images_para_ = os.path.join(image_para_root, imageFileYear, imageFileName)
                #images_para_ = images_path_.replace("Y_TC_ESTIMATE/SelectPic", "Y_TC_ESTIMATE/ALL/Parameter/MSW")
                #images_para_ = images_path_.replace("Y_TC_ESTIMATE_HUR/SelectPic/IR_TYPE", "Y_TC_ESTIMATE_HUR/ALL/Parameter/Type")
                #images_para_ = images_path_.replace("Y_TC_ESTIMATE_HUR/SelectPic/IR_MSW/","Y_TC_ESTIMATE_HUR/ALL/Parameter/MSW/")
                #images_para_ = images_path_.replace("Y_TC_ESTIMATE/SelectPic/IR_MSLP","Y_TC_ESTIMATE/ALL/Parameter/MSLP")

                images_para_ = images_para_.replace(".png\n", ".txt")
                with open(images_para_, "r") as f1:
                    TC_TYPE = f1.readline()

                    # Do not use Type == -1
                    if float(TC_TYPE)<0:
                        continue

                    images_path.append(images_path_.strip())
                    ALL_TC_TYPE.append(float(TC_TYPE))

        #b = np.unique(ALL_TC_TYPE)
        #hist, bins = np.histogram(ALL_TC_TYPE, bins=7, range=(0, 6))

        self.RootImageDir = "/media/dell/564C2A944C2A6F45/DataSet/Y_TC_ESTIMATE_HUR/SelectPic/IR_TYPE"
        self.RootParameterDir = "/media/dell/564C2A944C2A6F45/DataSet/Y_TC_ESTIMATE_HUR/ALL/Parameter/Type"

        #self.RootImageDir = "/mnt/564C2A944C2A6F45/DataSet/Y_TC_ESTIMATE/SelectPic/IR_MSW"
        #self.RootParameterDir = "/mnt/564C2A944C2A6F45/DataSet/Y_TC_ESTIMATE/ALL/Parameter/MSW"

        #self.RootImageDir = "/mnt/564C2A944C2A6F45/DataSet/Y_TC_ESTIMATE/SelectPic/IR_MSLP"
        #self.RootParameterDir = "/mnt/564C2A944C2A6F45/DataSet/Y_TC_ESTIMATE/ALL/Parameter/MSLP"

        self.images_path = images_path
        self.images_class = ALL_TC_TYPE
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def Usual_GetItem(self, item):
        img = Image.open(self.images_path[item])

        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def FANet_GetItem(self, item):
        img = Image.open(self.images_path[item])

        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        # Perform Polar Transform: [3, 224, 224]
        Polar_Image = np.asarray(img)
        # [3, 224, 224] ---> [224, 224, 3]
        Polar_Image = np.transpose(Polar_Image, [1, 2, 0])
        # Polar Transform:
        Polar_Image = warp_polar(Polar_Image, radius=112, multichannel=True)
        # plt.imshow(Polar_Image)
        # plt.show()

        Polar_Image = cv2.resize(Polar_Image, (224, 224))
        # [224, 224, 3] ---> [3, 224, 224]
        Polar_Image = np.transpose(Polar_Image, [2, 1, 0])
        Polar_Image = torch.from_numpy(Polar_Image)

        All_Image = torch.cat((img, Polar_Image), 0)

        return All_Image, label

    def __getitem__(self, item):

        return self.Usual_GetItem(item)
        #return self.FANet_GetItem(item)

        # img = Image.open(self.images_path[item])
        #
        # # RGB为彩色图片，L为灰度图片
        # if img.mode != 'RGB':
        #     raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        # label = self.images_class[item]
        #
        # if self.transform is not None:
        #     img = self.transform(img)
        #
        # # Perform Polar Transform: [3, 224, 224]
        # Polar_Image = np.asarray(img)
        # # [3, 224, 224] ---> [224, 224, 3]
        # Polar_Image = np.transpose(Polar_Image,[1,2,0])
        # # Polar Transform:
        # Polar_Image = warp_polar(Polar_Image, radius=112, multichannel=True)
        # #plt.imshow(Polar_Image)
        # #plt.show()
        #
        # Polar_Image = cv2.resize(Polar_Image, (224, 224))
        # # [224, 224, 3] ---> [3, 224, 224]
        # Polar_Image = np.transpose(Polar_Image, [2,1,0])
        # Polar_Image = torch.from_numpy(Polar_Image)
        #
        # All_Image = torch.cat((img, Polar_Image),0)
        #
        # return All_Image, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels, dtype=torch.float)

        return images, labels


class MyDataSetDeepTI(Dataset):
    """自定义数据集"""

    def __init__(self, Flag, transform=None):

        # Read From TxTFile
        train_data = []

        train_source = 'nasa_tropical_storm_competition_train_source'
        train_labels = 'nasa_tropical_storm_competition_train_labels'

        download_dir = Path("/home/chenhuanxin/datasets/NASA-tropical-storm")

        self.images_path = []
        self.images_class = []

        self.TrainImageList = []
        self.TestImageList = []

        self.TrainImageLabelList = []
        self.TestImageLabelList = []

        if os.path.isfile("AllData.txt"):
            with open('AllData.txt', 'rb') as file2:
                self.TrainImageList = pickle.load(file2)
                self.TrainImageLabelList = pickle.load(file2)
                self.TestImageList = pickle.load(file2)
                self.TestImageLabelList = pickle.load(file2)

        else:

            #if Flag == "Train":

            jpg_names = glob(str(download_dir / train_source / '**' / '*.jpg'))
            Count1 = 0
            Count2 = 0
            for jpg_path in jpg_names:
                # Count1 += 1
                # if Count1 > 4:
                #       continue

                self.TrainImageList.append(jpg_path)
                jpg_path = Path(jpg_path)

                # Get the IDs and file paths
                features_path = jpg_path.parent / 'features.json'
                image_id = '_'.join(jpg_path.parent.stem.rsplit('_', 3)[-2:])
                storm_id = image_id.split('_')[0]
                labels_path = str(jpg_path.parent / 'labels.json').replace(train_source, train_labels)

                # Load the features data
                with open(features_path) as src:
                    features_data = json.load(src)

                # Load the labels data
                with open(labels_path) as src:
                    labels_data = json.load(src)

                self.TrainImageLabelList.append(int(labels_data['wind_speed']))

            self.images_path = self.TrainImageList
            self.images_class = self.TrainImageLabelList

            #     train_data.append([
            #         image_id,
            #         storm_id,
            #         int(features_data['relative_time']),
            #         int(features_data['ocean']),
            #         int(labels_data['wind_speed'])
            #     ])
            #
            # train_df = pd.DataFrame(
            #     np.array(train_data),
            #     columns=['Image ID', 'Storm ID', 'Relative Time', 'Ocean', 'Wind Speed']
            # ).sort_values(by=['Image ID']).reset_index(drop=True)
            #
            # print(train_df.head())

            test_data = []

            test_source = 'nasa_tropical_storm_competition_test_source'
            test_labels = 'nasa_tropical_storm_competition_test_labels'

            #if Flag == "Val" or Flag == "Test":

            jpg_names = glob(str(download_dir / test_source / '**' / '*.jpg'))

            for jpg_path in jpg_names:
                # Count2 += 1
                # if Count2 > 4:
                #       continue

                self.TestImageList.append(jpg_path)
                jpg_path = Path(jpg_path)
                # Get the IDs and file paths
                features_path = jpg_path.parent / 'features.json'
                image_id = '_'.join(jpg_path.parent.stem.rsplit('_', 3)[-2:])
                storm_id = image_id.split('_')[0]

                labels_path = str(jpg_path.parent / 'labels.json').replace(test_source, test_labels)

                # Load the features data
                with open(features_path) as src:
                    features_data = json.load(src)

                with open(labels_path) as src:
                    labels_data = json.load(src)

                self.TestImageLabelList.append(int(labels_data['wind_speed']))

            self.images_path = self.TestImageList
            self.images_class = self.TestImageLabelList

            #AllTestLabel = np.array(self.TestImageLabelList)

            with open('AllData.txt', 'wb') as file:
                pickle.dump(self.TrainImageList, file)
                pickle.dump(self.TrainImageLabelList, file)
                pickle.dump(self.TestImageList, file)
                pickle.dump(self.TestImageLabelList, file)


        #AllTestLabelUnique = np.unique(AllTestLabel)

        #print(AllTestLabelUnique)
        #a = 1

        #     test_data.append([
        #         image_id,
        #         storm_id,
        #         int(features_data['relative_time']),
        #         int(features_data['ocean']),
        #         int(labels_data['wind_speed']),
        #     ])
        #
        # test_df = pd.DataFrame(
        #     np.array(test_data),
        #     columns=['Image ID', 'Storm ID', 'Relative Time', 'Ocean', 'Wind Speed']
        # ).sort_values(by=['Image ID']).reset_index(drop=True)
        #
        # print(test_df.head())

        self.transform = transform
        if Flag == "Train":
            self.images_path = self.TrainImageList
            self.images_class = self.TrainImageLabelList

        if Flag == "Val":
            self.images_path = self.TestImageList
            self.images_class = self.TestImageLabelList

        if Flag == "Test":
            self.images_path = self.TestImageList
            self.images_class = self.TestImageLabelList

    def __len__(self):
        return len(self.images_path)

    def Usual_GetItem(self, item):
        #img = Image.open(self.images_path[item])
        img = cv2.imread(self.images_path[item])
        img = PIL.Image.fromarray(img)

        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def FANet_GetItem(self, item):
        img = Image.open(self.images_path[item])

        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        # Perform Polar Transform: [3, 224, 224]
        Polar_Image = np.asarray(img)
        # [3, 224, 224] ---> [224, 224, 3]
        Polar_Image = np.transpose(Polar_Image, [1, 2, 0])
        # Polar Transform:
        Polar_Image = warp_polar(Polar_Image, radius=112, multichannel=True)
        # plt.imshow(Polar_Image)
        # plt.show()

        Polar_Image = cv2.resize(Polar_Image, (224, 224))
        # [224, 224, 3] ---> [3, 224, 224]
        Polar_Image = np.transpose(Polar_Image, [2, 1, 0])
        Polar_Image = torch.from_numpy(Polar_Image)

        All_Image = torch.cat((img, Polar_Image), 0)

        return All_Image, label

    def __getitem__(self, item):

        return self.Usual_GetItem(item)
        #return self.FANet_GetItem(item)

        # img = Image.open(self.images_path[item])
        #
        # # RGB为彩色图片，L为灰度图片
        # if img.mode != 'RGB':
        #     raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        # label = self.images_class[item]
        #
        # if self.transform is not None:
        #     img = self.transform(img)
        #
        # # Perform Polar Transform: [3, 224, 224]
        # Polar_Image = np.asarray(img)
        # # [3, 224, 224] ---> [224, 224, 3]
        # Polar_Image = np.transpose(Polar_Image,[1,2,0])
        # # Polar Transform:
        # Polar_Image = warp_polar(Polar_Image, radius=112, multichannel=True)
        # #plt.imshow(Polar_Image)
        # #plt.show()
        #
        # Polar_Image = cv2.resize(Polar_Image, (224, 224))
        # # [224, 224, 3] ---> [3, 224, 224]
        # Polar_Image = np.transpose(Polar_Image, [2,1,0])
        # Polar_Image = torch.from_numpy(Polar_Image)
        #
        # All_Image = torch.cat((img, Polar_Image),0)
        #
        # return All_Image, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels, dtype=torch.float)

        return images, labels

def Transform_WindSpeed_Classes(WindSpeed):
    # Transform into 7 classes
    # Tropical Depression  <= 33
    # Tropical Storm       34–64
    # Category 1           65–96
    # Category 2           97–110
    # Category 3           111–130
    # Category 4           131–155
    # Category 5           > 155

    if WindSpeed <= 33:
       WSclass = 1
       return WSclass
    if WindSpeed <= 64:
       WSclass = 2
       return WSclass
    if WindSpeed <= 96:
       WSclass = 3
       return WSclass
    if WindSpeed <= 110:
       WSclass = 4
       return WSclass
    if WindSpeed <= 130:
       WSclass = 5
       return WSclass
    if WindSpeed <= 155:
       WSclass = 6
       return WSclass
    else:
       WSclass = 7
       return WSclass

#-1 = Tropical depression (W<34)
#0 = Tropical storm [34<W<64]
#1 = Category 1 [64<=W<83]
#2 = Category 2 [83<=W<96]
#3 = Category 3 [96<=W<113]
#4 = Category 4 [113<=W<137]
#5 = Category 5 [W >= 137]
def Transform_WindSpeed_Classes2(WindSpeed):
    # Transform into 7 classes
    # -1 = Tropical depression (W<34)
    # 0 = Tropical storm [34<W<64]
    # 1 = Category 1 [64<=W<83]
    # 2 = Category 2 [83<=W<96]
    # 3 = Category 3 [96<=W<113]
    # 4 = Category 4 [113<=W<137]
    # 5 = Category 5 [W >= 137]

    if WindSpeed <= 33:
       WSclass = 1
       return WSclass
    if WindSpeed <= 63:
       WSclass = 2
       return WSclass
    if WindSpeed <= 82:
       WSclass = 3
       return WSclass
    if WindSpeed <= 96:
       WSclass = 4
       return WSclass
    if WindSpeed <= 112:
       WSclass = 5
       return WSclass
    if WindSpeed <= 136:
       WSclass = 6
       return WSclass
    else:
       WSclass = 7
       return WSclass



class MyDataSetDeepTI_5c(Dataset):
    """自定义数据集"""

    def __init__(self, Flag, transform=None):

        # Read From TxTFile
        train_data = []

        train_source = 'nasa_tropical_storm_competition_train_source'
        train_labels = 'nasa_tropical_storm_competition_train_labels'

        download_dir = Path("/media/dell/564C2A944C2A6F45/DataSet/DeepTI")

        self.images_path = []
        self.images_class = []

        self.TrainImageList = []
        self.TestImageList = []

        self.TrainImageLabelList = []
        self.TestImageLabelList = []
        self.alllist = []

        if os.path.isfile("AllData5c.txt"):
            with open('AllData5c.txt', 'rb') as file2:
                self.TrainImageList = pickle.load(file2)
                self.TrainImageLabelList = pickle.load(file2)
                self.TestImageList = pickle.load(file2)
                self.TestImageLabelList = pickle.load(file2)

        else:

            jpg_names = glob(str(download_dir / train_source / '**' / '*.jpg'))
            Count1 = 0
            Count2 = 0

            for jpg_path in jpg_names:
                #Count1 += 1
                #if Count1 > 10:
                #     continue

                self.TrainImageList.append(jpg_path)
                #showImagePath = cv2.imread(jpg_path)
                #print(showImagePath.shape)
                #cv2.imshow('image', showImagePath)  # 显示图片
                #cv2.waitKey(0)

                jpg_path = Path(jpg_path)
                # Get the IDs and file paths
                features_path = jpg_path.parent / 'features.json'
                image_id = '_'.join(jpg_path.parent.stem.rsplit('_', 3)[-2:])
                storm_id = image_id.split('_')[0]
                labels_path = str(jpg_path.parent / 'labels.json').replace(train_source, train_labels)

                # Load the features data
                with open(features_path) as src:
                    features_data = json.load(src)

                # Load the labels data
                with open(labels_path) as src:
                    labels_data = json.load(src)

                #self.TrainImageLabelList.append(int(labels_data['wind_speed']))
                self.TrainImageLabelList.append(Transform_WindSpeed_Classes2(int(labels_data['wind_speed'])))





            #print(np.max(self.TrainImageLabelList))
            #print(np.min(self.TrainImageLabelList))

            #hist(self.TrainImageLabelList, 8)
            #show()

            # sets = set(self.TrainImageLabelList)
            # dicts = {}
            # for item in sets:
            #     dicts.update({item:list.count(item)})
            # print(dicts)
            # results = Counter(self.TrainImageLabelList)
            # print(results)
            # a = 1


            # Transform into 7 classes
            #Tropical Depression  <= 33
            #Tropical Storm       34–64
            #Category 1           74–95
            #Category 2           96–110
            #Category 3           111–130
            #Category 4           131–155
            #Category 5           > 155

            #     train_data.append([
            #         image_id,
            #         storm_id,
            #         int(features_data['relative_time']),
            #         int(features_data['ocean']),
            #         int(labels_data['wind_speed'])
            #     ])
            #
            # train_df = pd.DataFrame(
            #     np.array(train_data),
            #     columns=['Image ID', 'Storm ID', 'Relative Time', 'Ocean', 'Wind Speed']
            # ).sort_values(by=['Image ID']).reset_index(drop=True)
            #
            # print(train_df.head())

            test_data = []

            test_source = 'nasa_tropical_storm_competition_test_source'
            test_labels = 'nasa_tropical_storm_competition_test_labels'

            jpg_names = glob(str(download_dir / test_source / '**' / '*.jpg'))

            for jpg_path in jpg_names:
                #Count2 += 1
                #if Count2 > 10:
                #     continue

                self.TestImageList.append(jpg_path)
                jpg_path = Path(jpg_path)
                # Get the IDs and file paths
                features_path = jpg_path.parent / 'features.json'
                image_id = '_'.join(jpg_path.parent.stem.rsplit('_', 3)[-2:])
                storm_id = image_id.split('_')[0]

                labels_path = str(jpg_path.parent / 'labels.json').replace(test_source, test_labels)

                # Load the features data
                with open(features_path) as src:
                    features_data = json.load(src)

                with open(labels_path) as src:
                    labels_data = json.load(src)

                #self.TestImageLabelList.append(int(labels_data['wind_speed']))
                self.TestImageLabelList.append(Transform_WindSpeed_Classes2(int(labels_data['wind_speed'])))

            with open('AllData5c.txt', 'wb') as file:
                pickle.dump(self.TrainImageList, file)
                pickle.dump(self.TrainImageLabelList, file)
                pickle.dump(self.TestImageList, file)
                pickle.dump(self.TestImageLabelList, file)

        # results = Counter(self.TestImageLabelList)
        # print(results)
        # a = 1
        #
        # self.alllist = self.TestImageLabelList + self.TrainImageLabelList
        # results = Counter(self.alllist)
        # print(results)

        #AllTestLabel = np.array(self.TestImageLabelList)
        #hist(self.TestImageLabelList, 128)
        #show()

        #AllTestLabelUnique = np.unique(AllTestLabel)

        #print(AllTestLabelUnique)
        #a = 1

        #     test_data.append([
        #         image_id,
        #         storm_id,
        #         int(features_data['relative_time']),
        #         int(features_data['ocean']),
        #         int(labels_data['wind_speed']),
        #     ])
        #
        # test_df = pd.DataFrame(
        #     np.array(test_data),
        #     columns=['Image ID', 'Storm ID', 'Relative Time', 'Ocean', 'Wind Speed']
        # ).sort_values(by=['Image ID']).reset_index(drop=True)
        #
        # print(test_df.head())

        self.transform = transform

        if Flag == "Train":
            self.images_path = self.TrainImageList
            self.images_class = self.TrainImageLabelList

        if Flag == "Val":
            self.images_path = self.TestImageList
            self.images_class = self.TestImageLabelList

        if Flag == "Test":
            self.images_path = self.TestImageList
            self.images_class = self.TestImageLabelList

    def __len__(self):
        return len(self.images_path)

    def Usual_GetItem(self, item):
        #img = Image.open(self.images_path[item])
        img = cv2.imread(self.images_path[item])
        img = PIL.Image.fromarray(img)

        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def FANet_GetItem(self, item):
        img = Image.open(self.images_path[item])

        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        # Perform Polar Transform: [3, 224, 224]
        Polar_Image = np.asarray(img)
        # [3, 224, 224] ---> [224, 224, 3]
        Polar_Image = np.transpose(Polar_Image, [1, 2, 0])
        # Polar Transform:
        Polar_Image = warp_polar(Polar_Image, radius=112, multichannel=True)
        # plt.imshow(Polar_Image)
        # plt.show()

        Polar_Image = cv2.resize(Polar_Image, (224, 224))
        # [224, 224, 3] ---> [3, 224, 224]
        Polar_Image = np.transpose(Polar_Image, [2, 1, 0])
        Polar_Image = torch.from_numpy(Polar_Image)

        All_Image = torch.cat((img, Polar_Image), 0)

        return All_Image, label

    def __getitem__(self, item):

        return self.Usual_GetItem(item)
        #return self.FANet_GetItem(item)

        # img = Image.open(self.images_path[item])
        #
        # # RGB为彩色图片，L为灰度图片
        # if img.mode != 'RGB':
        #     raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        # label = self.images_class[item]
        #
        # if self.transform is not None:
        #     img = self.transform(img)
        #
        # # Perform Polar Transform: [3, 224, 224]
        # Polar_Image = np.asarray(img)
        # # [3, 224, 224] ---> [224, 224, 3]
        # Polar_Image = np.transpose(Polar_Image,[1,2,0])
        # # Polar Transform:
        # Polar_Image = warp_polar(Polar_Image, radius=112, multichannel=True)
        # #plt.imshow(Polar_Image)
        # #plt.show()
        #
        # Polar_Image = cv2.resize(Polar_Image, (224, 224))
        # # [224, 224, 3] ---> [3, 224, 224]
        # Polar_Image = np.transpose(Polar_Image, [2,1,0])
        # Polar_Image = torch.from_numpy(Polar_Image)
        #
        # All_Image = torch.cat((img, Polar_Image),0)
        #
        # return All_Image, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels, dtype=torch.float)

        return images, labels
