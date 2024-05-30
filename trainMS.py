import argparse
import math
import shutil

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

from utils import progress_bar
from loss.spc import SupervisedContrastiveLoss
from loss.focalloss import FocalLoss
from data_augmentation.auto_augment import AutoAugment
from data_augmentation.duplicate_sample_transform import DuplicateSampleTransform

#from models.resnet_contrastive import get_resnet_ms
from models.origin_resnet import get_resnet_ms

from my_dataset import MyDataSetDeepTI, MyDataSetDeepTI_5c, MyDataSetTCIR, MySubset, MySubsetGeo_WS, MySubsetGeo , MySubset_WS
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
# # for smoothing
# from more_itertools import sort_together

rmse = lambda y_true, y_pred: math.sqrt(mean_squared_error(y_true, y_pred))

def mean_absolute_percentage_error(y_pred, y_true):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def symmetric_mean_absolute_precentage_error(y_pred, y_true):
    denominator = np.abs(y_true) + np.abs(y_pred)
    mask = denominator != 0
    return np.mean(2 * np.abs(y_pred[mask] - y_true[mask]) / denominator[mask])

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
        default=32,
        #default=1,
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
    #parser.add_argument("--auto-augment", default=True, type=bool)

    # focal contrasive alpha
    parser.add_argument("--alpha", default=1, type=float)

    parser.add_argument("--n_epochs_contrastive", default=75, type=int)
    #parser.add_argument("--n_epochs_cross_entropy", default=175, type=int)
    #parser.add_argument("--n_epochs_cross_entropy", default=300, type=int)
    #parser.add_argument("--n_epochs_cross_entropy", default=400, type=int)
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

    parser.add_argument('--weights', default="/home/chenhuanxin/models/resnet34-333f7ec4.pth", type=str, help='initial weights path')

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

    parser.add_argument('--use_test', action="store_true", default=False)

    parser.add_argument('--multi_mode', action="store_true", default=False)

    args = parser.parse_args()

    return args

def my_smooth(src):
    for idx in range(1, len(src) - 1):
        src[idx] = (src[idx - 1] + src[idx] + src[idx + 1]) / 3
    return src

def adjust_learning_rate(optimizer, epoch, loghandle, mode, args):
    """

    :param optimizer: torch.optim
    :param epoch: int
    :param mode: str
    :param args: argparse.Namespace
    :return: None
    """
    if mode == "contrastive":
        lr = args.lr_contrastive
        n_epochs = args.n_epochs_contrastive
    elif mode == "cross_entropy":
        lr = args.lr_cross_entropy
        n_epochs = args.n_epochs_cross_entropy
    else:
        lr = args.lr_cross_entropy
        n_epochs = args.n_epochs_cross_entropy
        #raise ValueError("Mode %s unknown" % mode)

    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / n_epochs)) / 2

    if args.step:
        n_steps_passed = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if n_steps_passed > 0:
            lr = lr * (args.lr_decay_rate ** n_steps_passed)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    loghandle.write("Adjusting Lr = " + str(lr) + "\n")
    loghandle.flush()

def train_contrastive(model, train_loader, criterion, optimizer, writer, args, loghandle, ckpt_save_path):
    """

    :param model: torch.nn.Module Model
    :param train_loader: torch.utils.data.DataLoader
    :param criterion: torch.nn.Module Loss
    :param optimizer: torch.optim
    :param writer: torch.utils.tensorboard.SummaryWriter
    :param args: argparse.Namespace
    :return: None
    """
    model.train()
    best_loss = float("inf")

    for epoch in range(args.n_epochs_contrastive):
        print("Epoch [%d/%d]" % (epoch + 1, args.n_epochs_contrastive))

        train_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = torch.cat(inputs)
            targets = targets.repeat(2)

            inputs, targets = inputs.to(args.device), targets.to(args.device)
            optimizer.zero_grad()

            projections = model.forward_constrative(inputs)
            loss = criterion(projections, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            writer.add_scalar(
                "Loss train | Supervised Contrastive",
                loss.item(),
                epoch * len(train_loader) + batch_idx,
            )

            progress_bar(
                batch_idx,
                len(train_loader),
                "Loss: %.3f " % (train_loss / (batch_idx + 1)),
            )

        avg_loss = train_loss / (batch_idx + 1)
        loghandle.write("Epoch  avg_loss= " + str(avg_loss) + "\n")
        loghandle.flush()
        # Only check every 10 epochs otherwise you will always save
        #if epoch % 10 == 0:
        if (train_loss / (batch_idx + 1)) < best_loss:
            print("Saving..")
            state = {
                "net": model.state_dict(),
                "avg_loss": avg_loss,
                "epoch": epoch,
            }

            if not os.path.isdir("checkpoint"):
                os.mkdir("checkpoint")

            if not os.path.isdir(ckpt_save_path):
                os.makedirs(ckpt_save_path)

            savepath = os.path.join(ckpt_save_path, "ckpt_contrastive.pth")
            torch.save(state, savepath)
            best_loss = avg_loss

        adjust_learning_rate(optimizer, epoch, loghandle, mode="contrastive", args=args)


def train_cross_entropy(model, train_loader, test_loader, criterion, optimizer, writer, args, loghandle, ckpt_save_path):
    """

    :param model: torch.nn.Module Model
    :param train_loader: torch.utils.data.DataLoader
    :param test_loader: torch.utils.data.DataLoader
    :param criterion: torch.nn.Module Loss
    :param optimizer: torch.optim
    :param writer: torch.utils.tensorboard.SummaryWriter
    :param args: argparse.Namespace
    :return:
    """

    for epoch in range(args.n_epochs_cross_entropy):  # loop over the dataset multiple times
        print("Epoch [%d/%d]" % (epoch + 1, args.n_epochs_cross_entropy))

        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets, loc_feats) in enumerate(train_loader):
            inputs, targets, loc_feats = inputs.to(args.device), targets.to(args.device), loc_feats.to(args.device)

            #print("inputs", inputs)
            #print("targets", targets)

            optimizer.zero_grad()
            #outputs = model(inputs)

            if args.model == "GoogleNet":
                outputs = model(inputs).logits
            else:
                outputs = model(inputs)
                #outputs = outputs[1]

            # o for contrastive, 1 for classification
            # outputs = outputs[1]
            # print(outputs)
            # print(targets)

            loss = criterion(outputs, targets)
            #print(targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)

            #print("predicted",predicted)

            total_batch = targets.size(0)
            correct_batch = predicted.eq(targets).sum().item()
            total += total_batch
            correct += correct_batch

            writer.add_scalar(
                "Loss train | Cross Entropy",
                loss.item(),
                epoch * len(train_loader) + batch_idx,
            )

            writer.add_scalar(
                "Accuracy train | Cross Entropy",
                correct_batch / total_batch,
                epoch * len(train_loader) + batch_idx,
            )

            progress_bar(
                batch_idx,
                len(train_loader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (
                    train_loss / (batch_idx + 1),
                    100.0 * correct / total,
                    correct,
                    total,
                ),
            )

        loghandle.write("CE Training Epoch = " + str(epoch) +
        " Loss train = " + str(loss.item()) + "Accuracy train = " + str(correct_batch / total_batch) + "\n")
        loghandle.flush()

        validation(epoch, model, test_loader, criterion, writer, args, loghandle, ckpt_save_path)

        adjust_learning_rate(optimizer, epoch, loghandle, mode='cross_entropy', args=args)

    print("Finished Training")

def train_any(model, train_loader, test_loader, criterion, optimizer, writer, args, loghandle, ckpt_save_path):
    """

    :param model: torch.nn.Module Model
    :param train_loader: torch.utils.data.DataLoader
    :param test_loader: torch.utils.data.DataLoader
    :param criterion: torch.nn.Module Loss
    :param optimizer: torch.optim
    :param writer: torch.utils.tensorboard.SummaryWriter
    :param args: argparse.Namespace
    :return:
    """

    criterion = torch.nn.MSELoss()
    #model.fc_class = nn.Linear(1536, 1)

    for epoch in range(args.n_epochs_cross_entropy):  # loop over the dataset multiple times
        print("Epoch [%d/%d]" % (epoch + 1, args.n_epochs_cross_entropy))

        model.train()
        train_loss = 0
        correct = 0
        total = 0
        if args.dataset == "TCIR_WS":
            for batch_idx, (inputs, targets, loc_feats, _, _) in enumerate(train_loader):
                inputs, targets, loc_feats = inputs.to(args.device), targets.to(args.device), loc_feats.to(args.device)

                #print("inputs", inputs)
                #print(inputs.size())

                #print("targets", targets)

                optimizer.zero_grad()
                #outputs = model(inputs)

                if args.model == "GoogleNet":
                    outputs = model(inputs).logits
                else:
                    #print(inputs)
                    outputs = model(inputs)
                    outputs = torch.squeeze(outputs)
                    #outputs = outputs[1]


                # o for contrastive, 1 for classification
                # outputs = outputs[1]
                #print(outputs)
                #print(targets)

                targets = targets.float()

                #print(outputs.size())
                #print(targets.size())

                loss = criterion(outputs, targets)
                #print(loss)

                #print(targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                #_, predicted = outputs.max(1)
                predicted = outputs

                #print("predicted",predicted)

                total_batch = targets.size(0)
                correct_batch = predicted.eq(targets).sum().item()
                total += total_batch
                correct += correct_batch

                writer.add_scalar(
                    "Loss train | Cross Entropy",
                    loss.item(),
                    epoch * len(train_loader) + batch_idx,
                )

                writer.add_scalar(
                    "Accuracy train | Cross Entropy",
                    correct_batch / total_batch,
                    epoch * len(train_loader) + batch_idx,
                )

                progress_bar(
                    batch_idx,
                    len(train_loader),
                    "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                    % (
                        train_loss / (batch_idx + 1),
                        100.0 * correct / total,
                        correct,
                        total,
                    ),
                )

        else:
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(args.device), targets.to(args.device)

                #print("inputs", inputs)
                #print(inputs.size())

                #print("targets", targets)

                optimizer.zero_grad()
                #outputs = model(inputs)

                if args.model == "GoogleNet":
                    outputs = model(inputs).logits
                else:
                    #print(inputs)
                    outputs = model(inputs)
                    outputs = torch.squeeze(outputs)
                    #outputs = outputs[1]


                # o for contrastive, 1 for classification
                # outputs = outputs[1]
                #print(outputs)
                #print(targets)

                targets = targets.float()

                #print(outputs.size())
                #print(targets.size())

                loss = criterion(outputs, targets)
                #print(loss)

                #print(targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                #_, predicted = outputs.max(1)
                predicted = outputs

                #print("predicted",predicted)

                total_batch = targets.size(0)
                correct_batch = predicted.eq(targets).sum().item()
                total += total_batch
                correct += correct_batch

                writer.add_scalar(
                    "Loss train | Cross Entropy",
                    loss.item(),
                    epoch * len(train_loader) + batch_idx,
                )

                writer.add_scalar(
                    "Accuracy train | Cross Entropy",
                    correct_batch / total_batch,
                    epoch * len(train_loader) + batch_idx,
                )

                progress_bar(
                    batch_idx,
                    len(train_loader),
                    "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                    % (
                        train_loss / (batch_idx + 1),
                        100.0 * correct / total,
                        correct,
                        total,
                    ),
                )

        loghandle.write("CE Training Epoch = " + str(epoch) +
        " Loss train = " + str(loss.item()) + "Accuracy train = " + str(correct_batch / total_batch) + "\n")
        loghandle.flush()

        validation_any(epoch, model, test_loader, criterion, writer, args, loghandle, ckpt_save_path)

        adjust_learning_rate(optimizer, epoch, loghandle, mode='cross_entropy', args=args)

    print("Finished Training")

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
    all_mse = []
    all_mae = []
    all_mape = []
    all_smape = []

    # for smoothing
    item_list = []

    with torch.no_grad():
        for batch_idx, (inputs, targets, loc_feats, time, id) in enumerate(test_loader):
            inputs, targets, loc_feats = inputs.to(args.device), targets.to(args.device), loc_feats.to(args.device)
            #outputs = model(inputs, Flag="Test")


            outputs = model(inputs)
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
            # print(type(predTrue[0]))
            tmp_list = []
            for i in range(targets.size(0)):
                tmp_list.append([predTrue[i], labelsTrue[i], id[i], time[i]])
            item_list.extend(tmp_list)

            cal_mse = mean_squared_error(labelsTrue, predTrue)
            all_mse.append(cal_mse)

            cal_mae = mean_absolute_error(labelsTrue, predTrue)
            all_mae.append(cal_mae)

            cal_mape = mean_absolute_percentage_error(predTrue, labelsTrue)
            all_mape.append(cal_mape)

            cal_smape = symmetric_mean_absolute_precentage_error(predTrue, labelsTrue)
            all_smape.append(cal_smape)

            cal_rmse = rmse(labelsTrue, predTrue)
            # cal_rmse = math.sqrt(cal_mse)
            All_RMSE.append(cal_rmse)

            progress_bar(
                batch_idx,
                len(test_loader),
                "Loss: %.3f | Acc: %.3f%% | RMSE: %.3f(%d/%d)" 
                % (
                    test_loss / (batch_idx + 1),
                    100.0 * correct / total,
                    np.mean(All_RMSE),
                    correct,
                    total,
                ),
            )

    # for smoothing
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

    # Save checkpoint.
    acc = 100.0 * correct / total
    myrmse = np.mean(All_RMSE) # TODO: 这里要加上MSE, MAE, MAPE, and SMAPE
    mymse = np.mean(all_mse)
    mymae = np.mean(all_mae)
    mymape = np.mean(all_mape)
    mysmape = np.mean(all_smape)
    loghandle.write("Epoch" + str(epoch) + " Tesing Accuracy: " + str(acc) + " Tesing rmse: " + str(myrmse) + " Smooth rmse: " + str(smooth_rmse) + " MSE: " + str(mymse) + " MAE: " + str(mymae) + " MAPE: " + str(mymape) + " SMAPE: " + str(mysmape) +"\n")
    loghandle.write("Best Accuracy: " + str(args.best_acc) + "Best RMSE: " + str(args.best_rmse) +"\n")

    loghandle.flush()


    print("[epoch {}] , accuracy: {}， RMSE: {}".format(epoch, acc, myrmse))

    writer.add_scalar("Accuracy validation | Cross Entropy", acc, epoch)

    if myrmse < args.best_rmse:
        print("Saving..")
        state = {
            "net": model.state_dict(),
            "rmse": myrmse,
            "epoch": epoch,
        }
        if not os.path.isdir("checkpoint"):
            os.mkdir("checkpoint")

        if not os.path.isdir(ckpt_save_path):
            os.makedirs(ckpt_save_path)

        args.best_rmse = myrmse

        Name = "ckpt_" + str(args.best_rmse) + ".pth"
        savepath = os.path.join(ckpt_save_path, Name)
        torch.save(state, savepath)

    if acc > args.best_acc:
        print("Saving..")
        state = {
            "net": model.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }
        if not os.path.isdir("checkpoint"):
            os.mkdir("checkpoint")

        if not os.path.isdir(ckpt_save_path):
            os.makedirs(ckpt_save_path)

        args.best_acc = acc

        Name = "ckpt_" + str(args.best_acc) + ".pth"
        savepath = os.path.join(ckpt_save_path, Name)
        torch.save(state, savepath)

def validation(epoch, model, test_loader, criterion, writer, args, loghandle, ckpt_save_path):
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

    with torch.no_grad():
        for batch_idx, (inputs, targets, loc_feats) in enumerate(test_loader):
            inputs, targets, loc_feats = inputs.to(args.device), targets.to(args.device), loc_feats.to(args.device)
            #outputs = model(inputs, Flag="Test")


            outputs = model(inputs)

            #loss = criterion(outputs, targets)
            #test_loss += loss.item()

            #print("outputs_size", outputs.size())
            #print("outputs", outputs)

            _, predicted = outputs.max(1)
            #pred = torch.max(outputs, dim=1)[1]

            #print("predicted",predicted)
            #print("pred",pred)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Calculate RMSE
            labelsTrue = targets.data.cpu().numpy()
            predTrue = predicted.data.cpu().numpy()

            cal_rmse = rmse(labelsTrue, predTrue)
            All_RMSE.append(cal_rmse)

            progress_bar(
                batch_idx,
                len(test_loader),
                "Loss: %.3f | Acc: %.3f%% | RMSE: %.3f(%d/%d)"
                % (
                    test_loss / (batch_idx + 1),
                    100.0 * correct / total,
                    np.mean(All_RMSE),
                    correct,
                    total,
                ),
            )

    # Save checkpoint.
    acc = 100.0 * correct / total
    myrmse = np.mean(All_RMSE)
    loghandle.write("Epoch" + str(epoch) + "Tesing Accuracy " + str(acc) + "Tesing rmse " + str(myrmse) + "\n")
    loghandle.write("Best Accuracy: " + str(args.best_acc) + "Best RMSE: " + str(args.best_rmse) +"\n")

    loghandle.flush()


    print("[epoch {}] , accuracy: {}， RMSE: {}".format(epoch, acc, myrmse))

    writer.add_scalar("Accuracy validation | Cross Entropy", acc, epoch)

    if myrmse < args.best_rmse:
        print("Saving..")
        state = {
            "net": model.state_dict(),
            "rmse": myrmse,
            "epoch": epoch,
        }
        if not os.path.isdir("checkpoint"):
            os.mkdir("checkpoint")

        if not os.path.isdir(ckpt_save_path):
            os.makedirs(ckpt_save_path)

        args.best_rmse = myrmse

        Name = "ckpt_" + str(args.best_rmse) + ".pth"
        savepath = os.path.join(ckpt_save_path, Name)
        torch.save(state, savepath)

    if acc > args.best_acc:
        print("Saving..")
        state = {
            "net": model.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }
        if not os.path.isdir("checkpoint"):
            os.mkdir("checkpoint")

        if not os.path.isdir(ckpt_save_path):
            os.makedirs(ckpt_save_path)

        args.best_acc = acc

        Name = "ckpt_" + str(args.best_acc) + ".pth"
        savepath = os.path.join(ckpt_save_path, Name)
        torch.save(state, savepath)

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

    sourcefile = "trainMS.py"
    destfile = os.path.join(SavedNowPath, sourcefile)
    shutil.copyfile(sourcefile, destfile)

    sourcefile = "my_dataset.py"
    destfile = os.path.join(SavedNowPath, sourcefile)
    shutil.copyfile(sourcefile, destfile)

    sourcefile = "models/resnet_contrastive.py"
    destfile = os.path.join(SavedNowPath, "resnet_contrastive.py")
    shutil.copyfile(sourcefile, destfile)

    sourcefile = "models/origin_resnet.py"
    destfile = os.path.join(SavedNowPath, "origin_resnet.py")
    shutil.copyfile(sourcefile, destfile)

    if args.dataset == "DeepTI":

        transform_train = [
            #transforms.RandomCrop(32, padding=4),
            #transforms.CenterCrop(128),
            transforms.Resize(224),
            #transforms.CenterCrop(64),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ]

        if args.auto_augment:
            transform_train.append(AutoAugment())

        transform_train.extend(
            [
                #transforms.Resize(224),
                #transforms.CenterCrop(64),
                transforms.ToTensor(),
                #transforms.Normalize(mean, std),
            ]
        )

        transform_train = transforms.Compose(transform_train)

        transform_test = transforms.Compose(
            [
                #transforms.CenterCrop(128),
                transforms.Resize(224),
                #transforms.CenterCrop(64),
                transforms.ToTensor(),
                #transforms.Normalize(mean, std),
            ]
        )

        #train_set = MyDataSetDeepTI("Train", transform=transform_train)
        train_set = MyDataSetDeepTI_5c("Train", transform=transform_train)
        #train_set = datasets.CIFAR10(root="~/data", train=True, download=True, transform=transform_train)

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )

        #test_set = datasets.CIFAR10(root="~/data", train=False, download=True, transform=transform_test)
        test_set = MyDataSetDeepTI_5c("Val", transform=transform_test)
        #test_set = MyDataSetDeepTI("Val", transform=transform_test)

        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        num_classes = 8

    if args.dataset == "DeepTI_WS":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        transform_train = [
            #transforms.RandomCrop(32, padding=4),
            #transforms.CenterCrop(128),
            transforms.Resize(224),
            #transforms.CenterCrop(64),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ]
        if args.auto_augment:
            transform_train.append(AutoAugment())

        transform_train.extend(
            [
                #transforms.Resize(224),
                #transforms.CenterCrop(64),
                transforms.ToTensor(),
                #transforms.Normalize(mean, std),
            ]
        )

        transform_train = transforms.Compose(transform_train)

        transform_test = transforms.Compose(
            [
                #transforms.CenterCrop(128),
                transforms.Resize(224),
                #transforms.CenterCrop(64),
                transforms.ToTensor(),
                #transforms.Normalize(mean, std),
            ]
        )

        train_set = MyDataSetDeepTI("Train", transform=transform_train)
        #train_set = MyDataSetDeepTI_5c("Train", transform=transform_train)
        #train_set = datasets.CIFAR10(root="~/data", train=True, download=True, transform=transform_train)

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )

        #test_set = datasets.CIFAR10(root="~/data", train=False, download=True, transform=transform_test)
        #test_set = MyDataSetDeepTI_5c("Val", transform=transform_test)
        test_set = MyDataSetDeepTI("Val", transform=transform_test)

        # mean = torch.zeros(3)
        # std = torch.zeros(3)
        # for X, _ in train_loader:
        #     for d in range(3):
        #         mean[d] += X[:, d, :, :].mean()
        #         std[d] += X[:, d, :, :].std()
        # mean.div_(len(train_set))
        # std.div_(len(train_set))
        # print(list(mean.numpy()), list(std.numpy()))
        #
        # a = 1

        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        num_classes = 200

    if args.dataset == "TCIR":

        train_path = '/home/chenhuanxin/datasets/TCIR-SPLT/TCIR-train.h5'
        if args.use_test:
            test_path = '/home/chenhuanxin/datasets/TCIR-SPLT/TCIR-test.h5'
        else:
            test_path = '/home/chenhuanxin/datasets/TCIR-SPLT/TCIR-val.h5'


        transform_train = [
            # transforms.RandomCrop(32, padding=4),
            # transforms.CenterCrop(128),
            # transforms.ToPILImage(),
            transforms.Resize(224),
            # transforms.CenterCrop(64),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ]

        if args.auto_augment:
            transform_train.append(AutoAugment())

        transform_train.extend(
            [
                # transforms.Resize(224),
                # transforms.CenterCrop(64),
                transforms.ToTensor(),
                # transforms.Normalize(mean, std),
            ]
        )

        transform_train = transforms.Compose(transform_train)

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

        train_subset = MyDataSetTCIR(train_path, args.multi_mode)
        train_set = MySubsetGeo(train_subset, transform=transform_train)

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )

        test_subset = MyDataSetTCIR(test_path, args.multi_mode)
        test_set = MySubsetGeo(test_subset, transform=transform_test)

        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        num_classes = 8

    if args.dataset == "TCIR_WS":
        train_path = '/home/chenhuanxin/datasets/TCIR-SPLT/TCIR-train.h5'
        if args.use_test:
            test_path = '/home/chenhuanxin/datasets/TCIR-SPLT/TCIR-test.h5'
        else:
            test_path = '/home/chenhuanxin/datasets/TCIR-SPLT/TCIR-val.h5'

        transform_train = [
            # transforms.RandomCrop(32, padding=4),
            # transforms.CenterCrop(128),
            # transforms.ToPILImage(),
            transforms.Resize(224),
            # transforms.CenterCrop(64),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ]

        if args.auto_augment:
            transform_train.append(AutoAugment())

        transform_train.extend(
            [
                # transforms.Resize(224),
                # transforms.CenterCrop(64),
                transforms.ToTensor(),
                # transforms.Normalize(mean, std),
            ]
        )

        transform_train = transforms.Compose(transform_train)

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

        train_subset = MyDataSetTCIR(train_path, args.multi_mode)
        #train_set = MySubsetGeo_WS(train_subset, transform=transform_train)
        train_set = MySubset_WS(train_subset, transform=transform_train)

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )

        test_subset = MyDataSetTCIR(test_path, args.multi_mode)
        #test_set = MySubsetGeo_WS(test_subset, transform=transform_test)
        test_set = MySubset_WS(test_subset, transform=transform_test)

        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        #num_classes = 200
        num_classes = 1

    #model = get_resnet_local_Global(args.model, num_classes)
    model = get_resnet_ms(args.model, num_classes)
    #model = resnet34(num_classes=num_classes)

    # Adding Pretrain model
    if os.path.exists(args.weights):
        # 我用的是python3.9，直接加载到GPU上会报错，issue: https://github.com/pytorch/vision/issues/3767
        # weights_dict = torch.load(args.weights, map_location=device)

        weights_dict = torch.load(args.weights, map_location="cpu")

        #print(weights_dict.keys())

        #model_dict = model.state_dict()
        #print(model_dict.keys())

        # Begin Several Layers
        # weights_dict[]
        new_state_dict = {}
        #for k, v in weights_dict['net'].items():
        for k, v in weights_dict.items():
            # Transfer TC classification into Regression
            if "linear" in k :
                continue

            if "layer" in k:

                NewLayerCaption_1 = k.replace("layer", "layer1_")
                NewLayerCaption_2 = k.replace("layer", "layer2_")
                NewLayerCaption_3 = k.replace("layer", "layer3_")

                new_state_dict[NewLayerCaption_1] = v
                new_state_dict[NewLayerCaption_2] = v
                new_state_dict[NewLayerCaption_3] = v

                continue
                #new_state_dict[k[7:]] = v

            if "bn1" in k :
                NewLayerCaption_1 = k.replace("bn1", "bn2")
                NewLayerCaption_2 = k.replace("bn1", "bn3")

                new_state_dict[k] = v
                new_state_dict[NewLayerCaption_1] = v
                new_state_dict[NewLayerCaption_2] = v

                continue

            if "conv1" in k:
                NewLayerCaption_1 = k.replace("conv1", "conv2")
                NewLayerCaption_2 = k.replace("conv1", "conv3")

                new_state_dict[k] = v
                new_state_dict[NewLayerCaption_1] = v
                new_state_dict[NewLayerCaption_2] = v

            #new_state_dict[k[7:]] = v

        #print(new_state_dict.keys())

        model.load_state_dict(new_state_dict, strict=False)
        #model.load_state_dict(weights_dict['net'], strict=True)

        print("Successful using pretrain-weights.")
    else:
        print("not using pretrain-weights.")

    if torch.cuda.device_count() > 1 and args.multi_gpu:
        # 如果不用os.environ的话，GPU的可见数量仍然是包括所有GPU的数量
        # 但是使用的还是只用指定的device_ids的GPU

        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model, device_ids=[0,1,2,3])

    model = model.to(args.device)

    cudnn.benchmark = True

    LogFile = os.path.join(SavedNowPath, "log.txt")
    file_handle = open(LogFile, 'w')
    file_handle.write(str(args) + "\n")

    description = args.description
    file_handle.write(description + "\n")

    writer = SummaryWriter("logs")

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr_cross_entropy,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    criterion = nn.CrossEntropyLoss()
    criterion.to(args.device)

    args.best_acc = 0.0
    args.best_rmse = 100.0
    #train_cross_entropy(model, train_loader, test_loader, criterion, optimizer, writer, args, file_handle, ckpt_save_path)
    train_any(model, train_loader, test_loader, criterion, optimizer, writer, args, file_handle,
                         ckpt_save_path)

    file_handle.close()


if __name__ == "__main__":
    main()
