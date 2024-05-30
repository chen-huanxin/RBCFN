import argparse
from inspect import ArgSpec
import math
import shutil
from unittest import result

import numpy as np
import os
import torch
import datetime
from shutil import copyfile
from torch.backends import cudnn
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.transforms.functional import rotate, InterpolationMode

from utils import progress_bar
from loss.spc import SupervisedContrastiveLoss
from loss.focalloss import FocalLoss
from data_augmentation.auto_augment import AutoAugment
from data_augmentation.duplicate_sample_transform import DuplicateSampleTransform
from models.resnet_contrastive import get_resnet_contrastive
from models.origin_resnet import resnet34
from my_dataset import MySubset_WS, MyDataSetTCIR
import math
from sklearn.metrics import mean_squared_error
from sklearn import datasets
from sklearn.manifold import TSNE
from test.ShowConfusionMatrix import *
from models.origin_resnet_cp import get_resnet_ms
import time
from PIL import Image
import cv2
from torchsummary import summary

# import tensorflow as tf
# import tensorflow_addons as tfa

rmse = lambda y_true, y_pred: math.sqrt(mean_squared_error(y_true, y_pred))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="resnet32",
        #default="resnet56",
        #default="resnet50",
        #default="OriginRes50",
        #default="OriginRes34",
        #default="resnet101",
        #default="resnet110",
        #default="DenseNet121",
        # 'resnet32': resnet32_ms,
        # 'resnet50': resnet50_ms,
        # 'resnet101': resnet101_ms,
        # "OriginRes50": resnet50,
        # "OriginRes34": resnet34
        choices=[
            "resnet32",
            "resnet50",
            "resnet101",
            "OriginRes50",
            "OriginRes34"
        ],
        help="Model to use",
    )

    # DeepTI_WS wind speed regression
    # DeepTI    wind type classification

    parser.add_argument(
        "--dataset",
        #default="TCIR",
        default="TCIR_WS",
        #default="DeepTI_WS",
        choices=["DeepTI", "DeepTI_WS", "TCIR", "TCIR_WS"],
        help="dataset name",
    )

    parser.add_argument(
        "--subset",
        default="ATLN",
        choices=["ALL", "ATLN", "CPAC"],
        help="subset name",
    )

    parser.add_argument(
        "--training_mode",
        default="cross-entropy",
        #default="contrastive",
        #default="focal",
        #default="focal_contrastive",
        #default="ce_contrastive",
        choices=["contrastive", "cross-entropy", "focal", "focal_contrastive", "ce_contrastive"],
        help="Type of training use either a two steps contrastive then cross-entropy or \
                         just cross-entropy",
    )

    # 0.1, 0.25. 0.5 ...
    parser.add_argument(
        "--focal_alpla",
        # default=32,
        default=None,
        type=float,
        help="On the contrastive step this will be multiplied by two.",
    )

    # 0, 0.1 , 5
    parser.add_argument(
        "--focal_gamma",
        # default=32,
        default=0.25,
        type=float,
        help="On the contrastive step this will be multiplied by two.",
    )

    parser.add_argument(
        "--batch_size",
        # default=32,
        default=1,
        type=int,
        help="On the contrastive step this will be multiplied by two.",
    )

    parser.add_argument(
        "--description",
        # default=32,
        default="MultiScale",
        type=str,
        help="The description of the model",
    )

    parser.add_argument("--temperature", default=0.1, type=float, help="Constant for loss no thorough ")

    parser.add_argument("--auto-augment", default=False, type=bool)

    # focal contrasive alpha
    parser.add_argument("--alpha", default=1, type=float)

    parser.add_argument("--n_epochs_contrastive", default=75, type=int)
    parser.add_argument("--n_epochs_cross_entropy", default=100, type=int)

    # Train From Scratch
    # parser.add_argument("--lr_contrastive", default=1e-1, type=float)
    # parser.add_argument("--lr_cross_entropy", default=0.05, type=float)

    # Train From Pretrained
    parser.add_argument("--lr_contrastive", default=0.01, type=float)
    #parser.add_argument("--lr_cross_entropy", default=0.05, type=float)
    parser.add_argument("--lr_cross_entropy", default=0.001, type=float)

    #parser.add_argument('--weights', type=str,
    #                    default='/media/dell/564C2A944C2A6F45/LinuxCode/TyphoonEstimation/Pretrained/resnet50-19c8e357.pth',
    #                    help='initial weights path')

    # parser.add_argument('--weights', type=str,
    #                    default='/media/dell/564C2A944C2A6F45/LinuxCode/TyphoonEstimation/Pretrained/resnet34-333f7ec4.pth',
    #                    help='initial weights path')

    # parser.add_argument('--weights', type=str,
    #                     default='/media/dell/564C2A944C2A6F45/LinuxCode/TyphoonEstimation/Pretrained/resnet101-5d3b4d8f.pth',
    #                     help='initial weights path')

    # parser.add_argument('--weights', type=str,
    #                     default='/media/dell/564C2A944C2A6F45/Code/Supervised_contrastive_loss_pytorch-main/checkpoint/2022_04_26_11_57_25focal_contrastive/ckpt_71.87957725848976.pth',
    #                     help='initial weights path')

    parser.add_argument('--weights', default="checkpoint/2022_08_09_03_47_08cross-entropy/ckpt_9.624193373371858.pth", type=str, help='initial weights path')

    parser.add_argument("--cosine", default=False, type=bool, help="Check this to use cosine annealing instead of ")

    parser.add_argument("--step", default=True, type=bool, help="Check this to use step")

    parser.add_argument("--lr_decay_rate", type=float, default=0.1, help="Lr decay rate when cosine is false")

    parser.add_argument(
        "--lr_decay_epochs",
        type=list,
        default=[50, 75],
        #default=[75, 150],
        #default=[100, 200, 300],
        help="If cosine false at what epoch to decay lr with lr_decay_rate",
    )

    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for SGD")

    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum for SGD")

    parser.add_argument("--num_workers", default=4, type=int, help="number of workers for Dataloader")

    parser.add_argument('--gpu', type=int, default=0, help='using gpu')

    parser.add_argument('--multi_gpu', action="store_true", default=False)

    parser.add_argument('--multi_mode', action="store_true", default=False)

    parser.add_argument('--smooth', action="store_true", default=False)

    parser.add_argument('--rotation_blend', action="store_true", default=False)

    parser.add_argument('--blend_num', type=int, default=6, help='num of blending')

    args = parser.parse_args()

    return args

def my_smooth(src):
    for idx in range(1, len(src) - 1):
        src[idx] = (src[idx - 1] + src[idx] + src[idx + 1]) / 3
    return src

def crop_center(matrix, crop_width):
    print(matrix.shape[2])
    total_width = matrix.shape[2]
    start = total_width // 2 - crop_width // 2
    end = start + crop_width
    return matrix[:, :, start:end, start:end]

def rotation_blending(model, blending_num, images, loc_feats, args):
    sum_outputs = torch.zeros(images.size(0), 1).cuda()
    times = 0
    for angle in np.linspace(0, 360, blending_num, endpoint=False):
        rotated_image = rotate(images, angle,  InterpolationMode.BILINEAR, fill=0)
        # print(rotated_image.size())
        # input_image = crop_center(rotated_image, 64)
        # print(rotated_image.size())
        output = model(rotated_image.to(args.device))
        sum_outputs += output
        times += 1

    return sum_outputs / times


def validation_any(epoch, model, test_loader, criterion, writer, args, loghandle, ckpt_save_path):
    """

    :param epoch: int
    :param model: torch.nn.Module, Model
    :param test_loader: torch.utils.data.DataLoader
    :param criterion: torch.nn.Module, Loss
    :param writer: torch.utils.tensorboard.SummaryWriter
    :param args: argparse.Namespace
    :return:
    """

    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    # Calculate RMSE
    All_RMSE = []

    # for smoothing
    item_list = []

    with torch.no_grad():
        # for batch_idx, (inputs, targets, loc_feats, Time, id) in enumerate(test_loader):
        dataiter = iter(test_loader)
        inputs, targets, loc_feats, Time, id = dataiter.next()
        targets, loc_feats = targets.to(args.device), loc_feats.to(args.device)

        if args.rotation_blend:
            outputs = rotation_blending(model, 6, inputs, loc_feats, args)
        else:
            inputs = inputs.to(args.device)  
            T1 = time.process_time()          
            outputs = model(inputs)
            T2 = time.process_time()
            print('time %f\n' % ((T2 - T1) * 1000))
            summary(model, (3, 224, 224))


        outputs = torch.squeeze(outputs)
        
        #loss = criterion(outputs, targets)
        #test_loss += loss.item()

        #print("outputs_size", outputs.size())
        #print("outputs", outputs)

        #_, predicted = outputs.max(1)
        #pred = torch.max(outputs, dim=1)[1]
        predicted = outputs

        #print("predicted",predicted)
        #print("pred",pred)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Calculate RMSE
        labelsTrue = targets.data.cpu().numpy()
        predTrue = predicted.data.cpu().numpy()

        # for smoothing
        if args.smooth:
            tmp_list = []
            for i in range(targets.size(0)):
                tmp_list.append([predTrue[i], labelsTrue[i], id[i], Time[i]])
            item_list.extend(tmp_list)

            cal_rmse = rmse(labelsTrue, predTrue)
            All_RMSE.append(cal_rmse)

            progress_bar(
                0,
                len(test_loader),
                "Loss: %.3f | Acc: %.3f%% | RMSE: %.3f(%d/%d)"
                % (
                    test_loss / (0 + 1),
                    100.0 * correct / total,
                    np.mean(All_RMSE),
                    correct,
                    total,
                ),
            )

    acc = 100.0 * correct / total
    myrmse = np.mean(All_RMSE)

    # for smoothing
    if args.smooth:
        sorted(item_list, key=lambda x: x[3])
        sorted(item_list, key=lambda x: x[2])

        pred_sort = []
        label_sort = []
        for item in item_list:
            pred_sort.append(item[0])
            label_sort.append(item[1])

        # # 利用卷积的方法做窗口大小为3的smooth
        # # numpy.convolve(a, v, mode='full')
        # # mode可能的三种取值情况：
        # # 'full'　默认值，返回每一个卷积值，长度是N+M-1,在卷积的边缘处，信号不重叠，存在边际效应。
        # # 'same'　返回的数组长度为max(M, N),边际效应依旧存在。
        # # 'valid' 　返回的数组长度为max(M,N)-min(M,N)+1,此时返回的是完全重叠的点。边缘的点无效
        # #
        # pred_sort = np.convolve(pred_sort, np.ones((3,))/3, mode='same')
        pred_sort = my_smooth(pred_sort)
        smooth_rmse = rmse(pred_sort, label_sort)

        loghandle.write("Epoch" + str(epoch) + "Tesing Accuracy " + str(acc) + "Tesing rmse " + str(myrmse) + "Smooth rmse " + str(smooth_rmse) + "\n")
    else:
        loghandle.write("Epoch" + str(epoch) + "Tesing Accuracy " + str(acc) + "Tesing rmse " + str(myrmse) + "\n")
    # loghandle.write("Best Accuracy: " + str(args.best_acc) + "Best RMSE: " + str(args.best_rmse) +"\n")

    loghandle.flush()

    print("[epoch {}], accuracy: {},  RMSE: {}".format(epoch, acc, myrmse))

    writer.add_scalar("Accuracy validation | Cross Entropy", acc, epoch)

def main():

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    args = parse_args()

    # #device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(args.gpu)
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.device = device

    if not os.path.isdir("logs"):
        os.makedirs("logs")

    now_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    NowMode = args.training_mode

    ckpt_save_path = os.path.join(os.getcwd(), "checkpoint", now_time + NowMode)
    SavedNowPath = os.path.join(os.getcwd(), "logs", now_time + NowMode)

    os.makedirs(SavedNowPath)

    # if args.dataset == "TCIR_WS":
    test_path = '/home/chenhuanxin/datasets/TCIR-SPLT/TCIR-test.h5'

    transform_test = transforms.Compose(
            [
                # transforms.CenterCrop(128),
                # transforms.ToPILImage(),
                transforms.Resize(224),
                # transforms.CenterCrop(64),
                transforms.ToTensor(),
                # transforms.Normalize(mean, std),
            ]
        )

    test_subset = MyDataSetTCIR(test_path, args.multi_mode)
    test_set = MySubset_WS(test_subset, transform=transform_test)
    print(len(test_set))
    input()
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    num_classes = 1
    model = get_resnet_ms(args.model, num_classes)
    if args.model == "OriginRes34":
        args.weights = "checkpoint/2022_08_10_02_25_55cross-entropy/ckpt_10.46196023958722.pth"
    weights_dict = torch.load(args.weights, map_location="cpu")
    new_state_dict = {}

    for k, v in weights_dict['net'].items():
        if "module" in k:
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=True)
    model = model.to(args.device)
    print("Successful using pretrain-weights.")

    LogFile = os.path.join(SavedNowPath, "log.txt")
    loghandle = open(LogFile, 'w')
    loghandle.write(str(args) + "\n")

    description = args.description
    loghandle.write(description + "\n")

    writer = SummaryWriter("logs")

    criterion = nn.CrossEntropyLoss()
    criterion.to(args.device)

    validation_any(0, model, test_loader, criterion, writer, args, loghandle, ckpt_save_path)
    loghandle.close()

if __name__ == "__main__":
    main()
