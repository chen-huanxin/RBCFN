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
from Evison import Display, show_network

from utils import progress_bar
from loss.spc import SupervisedContrastiveLoss
from loss.focalloss import FocalLoss
from data_augmentation.auto_augment import AutoAugment
from data_augmentation.duplicate_sample_transform import DuplicateSampleTransform
from models.resnet_contrastive import get_resnet_contrastive
from models.origin_resnet import resnet34
from my_dataset import MyDataSetDeepTI, MyDataSetDeepTI_5c
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn.manifold import TSNE
from test.ShowConfusionMatrix import *

import matplotlib.pyplot as plt
from time import time
from PIL import Image
import cv2

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

rmse = lambda y_true, y_pred: math.sqrt(mean_squared_error(y_true, y_pred))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        #default="resnet32",
        default="resnet32",
        choices=[
            "resnet20",
            "resnet32",
            "resnet44",
            "resnet56",
            "resnet110",
            "resnet1202",
        ],
        help="Model to use",
    )

    # DeepTI_WS wind speed regression
    # DeepTI    wind type classification

    parser.add_argument(
        "--dataset",
        default="DeepTI",
        #default="DeepTI_WS",
        choices=["DeepTI", "DeepTI_WS"],
        help="dataset name",
    )

    parser.add_argument(
        "--training_mode",
        #default="cross-entropy",
        #default="contrastive",
        #default="focal",
        default="focal_contrastive",
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
        default=2,
        type=float,
        help="On the contrastive step this will be multiplied by two.",
    )

    parser.add_argument(
        "--batch_size",
        default=32,
        #default=64,
        type=int,
        help="On the contrastive step this will be multiplied by two.",
    )

    parser.add_argument("--temperature", default=0.1, type=float, help="Constant for loss no thorough ")

    parser.add_argument("--auto-augment", default=False, type=bool)
    #parser.add_argument("--auto-augment", default=True, type=bool)

    # focal contrasive alpha
    parser.add_argument("--alpha", default=1, type=float)

    parser.add_argument("--n_epochs_contrastive", default=75, type=int)
    #parser.add_argument("--n_epochs_cross_entropy", default=175, type=int)
    #parser.add_argument("--n_epochs_cross_entropy", default=300, type=int)
    parser.add_argument("--n_epochs_cross_entropy", default=100, type=int)

    parser.add_argument("--lr_contrastive", default=1e-1, type=float)
    parser.add_argument("--lr_cross_entropy", default=0.05, type=float)

    # parser.add_argument('--weights', type=str,
    #                     default='/media/dell/564C2A944C2A6F45/LinuxCode/TyphoonEstimation/Pretrained/resnet50-19c8e357.pth',
    #                     help='initial weights path')

    # parser.add_argument('--weights', type=str,
    #                     default='/media/dell/564C2A944C2A6F45/LinuxCode/TyphoonEstimation/Pretrained/resnet34-333f7ec4.pth',
    #                     help='initial weights path')


    #parser.add_argument('--weights', default="/media/dell/564C2A944C2A6F45/Code/Supervised_contrastive_loss_pytorch-main/checkpoint/2022_04_19_16_25_31cross-entropy/ckpt_cross_entropy_69.75009577033148.pth", type=str, help='initial weights path')

    # parser.add_argument('--weights',
    #                     default="/media/dell/564C2A944C2A6F45/Code/Supervised_contrastive_loss_pytorch-main/checkpoint/2022_04_26_11_57_25focal_contrastive/ckpt_71.87957725848976.pth",
    #                     type=str, help='initial weights path')

    # parser.add_argument('--weights',
    #                     default="/media/dell/564C2A944C2A6F45/Code/Supervised_contrastive_loss_pytorch-main/checkpoint/2022_04_19_16_25_31cross-entropy/ckpt_cross_entropy_67.90905198638934.pth",
    #                     type=str, help='initial weights path')

    # parser.add_argument('--weights',
    #                     default="/media/dell/564C2A944C2A6F45/Code/Supervised_contrastive_loss_pytorch-main/checkpoint/2022_04_16_18_37_54focal_contrastive/ckpt_cross_entropy_70.88807265024676.pth",
    #                     type=str, help='initial weights path')

    parser.add_argument('--weights',
                        default="/media/dell/564C2A944C2A6F45/Code/Supervised_contrastive_loss_pytorch-main/checkpoint/2022_04_19_16_25_31cross-entropy/ckpt_cross_entropy_69.16420668364243.pth",
                        type=str, help='initial weights path')

    # parser.add_argument('--weights',
    #                     default="/media/dell/564C2A944C2A6F45/Code/Supervised_contrastive_loss_pytorch-main/checkpoint/2022_04_26_11_57_25focal_contrastive/ckpt_71.87957725848976.pth",
    #                     type=str, help='initial weights path')

    parser.add_argument("--cosine", default=False, type=bool, help="Check this to use cosine annealing instead of ")
    parser.add_argument("--step", default=True, type=bool, help="Check this to use step")

    parser.add_argument("--lr_decay_rate", type=float, default=0.1, help="Lr decay rate when cosine is false")

    parser.add_argument(
        "--lr_decay_epochs",
        type=list,
        default=[50, 75],
        #default=[75, 150],
        #default=[100, 200, 250],
        help="If cosine false at what epoch to decay lr with lr_decay_rate",
    )

    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for SGD")
    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum for SGD")

    parser.add_argument("--num_workers", default=4, type=int, help="number of workers for Dataloader")

    args = parser.parse_args()

    return args


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    names = ["Tropical depression", "Tropical storm",
             "Category 1", "Category 2", "Category 3",
             "Category 4", "Category 5"]

    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1],',',
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


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
    All_labelsTrue = []
    ALL_predTrue = []

    AllFeaturesCount = 1
    AllFeatures=[]

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):

            AllFeaturesCount += 1

            # if AllFeaturesCount > 10:
            #     continue
            inputs, targets = inputs.to(args.device), targets.to(args.device)

            outputs = model(inputs, Flag="Test")

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

            #print(labelsTrue)
            #print(predTrue)

            All_labelsTrue.extend(labelsTrue)
            ALL_predTrue.extend(predTrue)

            # T-Sne:
            Tsne = 0
            if Tsne == 1:

                tsneFeature = model.tsne_show
                #print(tsneFeature.size())
                tsneFeature = tsneFeature.data.cpu().numpy()
                AllFeatures.extend(tsneFeature.reshape(tsneFeature.shape[0],-1))

            #my_confusion = metrics.confusion_matrix(SelectOut, SelectGT).astype(np.float32)
            #meanIU, Acc, Se, Sp, IU = calculate_Accuracy(my_confusion)
            #Auc = roc_auc_score(tmp_gt, y_pred)
            #AUC.append(Auc)

            #Tropical
            #depression(W < 34)
            # 0 = Tropical storm [34<W<64]
            # 1 = Category 1 [64<=W<83]
            # 2 = Category 2 [83<=W<96]
            # 3 = Category 3 [96<=W<113]
            # 4 = Category 4 [113<=W<137]
            # 5 = Category 5 [W >= 137]

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


    showtsne = 0
    if showtsne==1:

        print('Computing t-SNE embedding')
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        t0 = time()
        AllFeatures = np.array(AllFeatures)
        print(AllFeatures.shape)
        result = tsne.fit_transform(AllFeatures)

        fig = plot_embedding(result, All_labelsTrue, 'T-SNE embedding of the logits using hybrid loss')

        plt.show()
        a = 1

    showConfusionMatrix = 0
    if showConfusionMatrix == 1:
        fig, ax = plt.subplots()

        vegetables = ["Tropical depression", "Tropical storm",
             "Category 1", "Category 2", "Category 3",
             "Category 4", "Category 5"]
        farmers = ["Tropical depression", "Tropical storm",
             "Category 1", "Category 2", "Category 3",
             "Category 4", "Category 5"]

        All_labelsTrue = np.array(All_labelsTrue)
        ALL_predTrue = np.array(ALL_predTrue)

        my_confusion = confusion_matrix(All_labelsTrue - 1, ALL_predTrue - 1)

        im, cbar = heatmap(my_confusion, vegetables, farmers, ax=ax,
                           cmap="YlGn", cbarlabel="Counts")
        texts = annotate_heatmap(im, valfmt="{x}")

        fig.tight_layout()

        plt.show()
        a = 1

    classificationR = 0
    if classificationR == 1:

        EMOS = ['Tropical depression', 'Tropical storm', 'Category 1', 'Category 2', 'Category 3', 'Category 4', 'Category 5']
        #EMOS = ['Tropical depression', 'Tropical storm']
        NUM_EMO = len(EMOS)

        All_labelsTrue= np.array(All_labelsTrue)
        ALL_predTrue = np.array(ALL_predTrue)

        print(All_labelsTrue.shape)
        print(ALL_predTrue.shape)

        #y_true = [0, 1, 2, 2, 2]
        #y_pred = [0, 0, 2, 2, 1]
        #target_names = ['class 0', 'class 1', 'class 2']

        # print(classification_report(y_true, y_pred, target_names=target_names, digits=4, labels=[0,1,2]))
        print(classification_report(All_labelsTrue-1, ALL_predTrue-1, target_names=EMOS, digits=4, labels=[0,1,2,3,4,5,6,7]))

        # Save checkpoint.
        acc = 100.0 * correct / total
        print("Acc: %.3f%%"% acc)
        myrmse = np.mean(All_RMSE)


def main():

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    args = parse_args()

    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    args.device = device
    # model = get_resnet_contrastive(args.model, num_classes)
    # model = resnet34(num_classes=num_classes)
    ckpt_save_path = 0

    if args.dataset == "DeepTI":

        transform_test = transforms.Compose(
            [
                #transforms.CenterCrop(128),
                transforms.Resize(224),
                #transforms.CenterCrop(64),
                transforms.ToTensor(),
                #transforms.Normalize(mean, std),
            ]
        )

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
        #train_set = MyDataSetDeepTI_5c("Train", transform=transform_train)
        #train_set = datasets.CIFAR10(root="~/data", train=True, download=True, transform=transform_train)

        # train_loader = torch.utils.data.DataLoader(
        #     train_set,
        #     batch_size=args.batch_size,
        #     shuffle=True,
        #     num_workers=args.num_workers,
        # )

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

    model = get_resnet_contrastive(args.model, num_classes)

    if os.path.exists(args.weights):
        weights_dict = torch.load(args.weights, map_location=device)

        NowState_dict = model.state_dict()
        # print(NowState_dict)
        # weights_dict = torch.load(args.weights)
        # modelkeys = model.state_dict().keys()
        # load_weights_dict = {k: v for k, v in weights_dict.items()
        #                      if k in model.state_dict().keys() and model.state_dict()[k].numel() == v.numel()}
        #
        new_state_dict = {}

        for k, v in weights_dict['net'].items():
            if "module" in k:
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict, strict=True)

        #model.load_state_dict(weights_dict['net'], strict=True)

        # model.load_state_dict(weights_dict, strict=True)

        print("Successful using pretrain-weights.")

    else:
        print("not using pretrain-weights.")

    # if torch.cuda.device_count() > 1:
    #     # 如果不用os.environ的话，GPU的可见数量仍然是包括所有GPU的数量
    #     # 但是使用的还是只用指定的device_ids的GPU
    #
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #     model = nn.DataParallel(model, device_ids=[0, 1])

    model = model.to(args.device)

    # For Evison

    # show which layer we can visualize
    UseEvison = 0

    if UseEvison == 1:
        show_network(model)
        visualized_layer = 'contrastive_hidden_layer'
        display = Display(model, visualized_layer, img_size=(224, 224))

        ImagePath = "/media/dell/564C2A944C2A6F45/DataSet/DeepTI/nasa_tropical_storm_competition_train_source"
        AllFile = os.listdir(ImagePath)
        count = 0
        for eachfile in AllFile:
            count += 1
            if count < 2:
                continue

            full_file_path = os.path.join(ImagePath, eachfile)
            # AllFile2 = os.listdir(full_file_path)
            ImagePath_ = os.path.join(full_file_path, "image.jpg")
            # ReadedImage = cv2.imread(ImagePath_)

            myimage = Image.open(ImagePath_).resize((224, 224))
            imrgb = Image.merge("RGB", (myimage, myimage, myimage))
            display.save(imrgb)
            a = 1

    UseCam = 1
    if UseCam == 1:
        target_layers = [model.layer3[-1]]
        #target_layers = [model.linear]
        model.SetFlag("Test")
        ImagePath = "/media/dell/564C2A944C2A6F45/DataSet/DeepTI/nasa_tropical_storm_competition_test_source"
        #savepath = "/media/dell/564C2A944C2A6F45/Code/Supervised_contrastive_loss_pytorch-main/visualization/Hybrid2"
        savepath = "/media/dell/564C2A944C2A6F45/Code/Supervised_contrastive_loss_pytorch-main/visualization/ce2"

        AllFile = os.listdir(ImagePath)
        count = 0
        for eachfile in AllFile:
            count += 1
            if count < 2:
                continue

            print("Handleing %s"%eachfile)
            saveImageName = os.path.join(savepath,eachfile + ".png")
            full_file_path = os.path.join(ImagePath, eachfile)
            # AllFile2 = os.listdir(full_file_path)
            ImagePath_ = os.path.join(full_file_path, "image.jpg")
            ReadedImage = cv2.imread(ImagePath_)
            ReadedImage = cv2.resize(ReadedImage, (224,224))
            #myimage = Image.open(ImagePath_).resize((224, 224))

            #imrgb = Image.merge("RGB", (myimage, myimage, myimage))
            imrgb = ReadedImage
            # [N, C, H, W]
            imrgb = Image.fromarray(imrgb)
            img_tensor = transform_test(imrgb)
            # expand batch dimension
            input_tensor = torch.unsqueeze(img_tensor, dim=0)

            cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
            target_category = num_classes  # tabby, tabby cat
            # target_category = 254  # pug, pug-dog

            grayscale_cam = cam(input_tensor=input_tensor)

            grayscale_cam = grayscale_cam[0, :]

            imrgb = np.array(imrgb)

            visualization = show_cam_on_image(imrgb.astype(dtype=np.float32) / 255.,
                                              grayscale_cam,
                                              use_rgb=True)

            visualization = cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB)
            cv2.imwrite(saveImageName, visualization)

            # visualization = show_cam_on_image(imrgb,
            #                                   grayscale_cam,
            #                                   use_rgb=True)

            #plt.imshow(visualization)
            #plt.savefig(saveImageName)

            #plt.show()
            #plt.pause()

    cudnn.benchmark = True

    #LogFile = os.path.join(SavedNowPath, "log.txt")
    file_handle = 0

    # file_handle = open(LogFile, 'w')
    # file_handle.write(str(args) + "\n")

    writer = 0

    if args.training_mode == "contrastive":

        model = model.to(args.device)

        criterion = nn.CrossEntropyLoss()
        criterion.to(args.device)

        args.best_acc = 0.0
        args.best_rmse = 100.0
        validation(1, model, test_loader, criterion, writer, args, file_handle, ckpt_save_path)

    elif args.training_mode == "focal":

        focal_alpha = args.focal_alpla
        focal_gamma = args.focal_gamma

        # criterion1 = FocalLoss(gamma=0)
        criterion = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        criterion.to(args.device)

        args.best_acc = 0.0
        args.best_rmse = 100.0
        # train_cross_entropy(model, train_loader, test_loader, criterion, optimizer, writer, args, file_handle,
        #                     ckpt_save_path)
        validation(1, model, test_loader, criterion, writer, args, file_handle, ckpt_save_path)

    elif args.training_mode == "ce_contrastive":

        model = model.to(args.device)


        #focal_alpha = args.focal_alpla
        #focal_gamma = args.focal_gamma

        #criterion1 = FocalLoss(gamma=0)
        #criterion1 = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)

        criterion1 = nn.CrossEntropyLoss()

        args.best_acc = 0.0
        args.best_rmse = 100.0
        # train_focal_contrastive(model, train_loader_contrastive, test_loader, criterion1, criterion2, optimizer, writer, args, file_handle,
        #                     ckpt_save_path)
        validation(1, model, test_loader, criterion1, writer, args, file_handle, ckpt_save_path)

    elif args.training_mode == "focal_contrastive":


        model = model.to(args.device)


        focal_alpha = args.focal_alpla
        focal_gamma = args.focal_gamma

        #criterion1 = FocalLoss(gamma=0)
        criterion1 = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)


        args.best_acc = 0.0
        args.best_rmse = 100.0
        # train_focal_contrastive(model, train_loader_contrastive, test_loader, criterion1, criterion2, optimizer, writer, args, file_handle,
        #                     ckpt_save_path)
        validation(1, model, test_loader, criterion1, writer, args, file_handle, ckpt_save_path)

    else:

        criterion = nn.CrossEntropyLoss()
        criterion.to(args.device)

        args.best_acc = 0.0
        args.best_rmse = 100.0

        validation(1, model, test_loader, criterion, writer, args, file_handle, ckpt_save_path)


if __name__ == "__main__":
    main()
