'''
Paper: https://arxiv.org/abs/1512.03385
Original code taken from: https://github.com/akamaster/pytorch_resnet_cifar10

Note that this code has been modified to have a contrastive approach. By design there are two added function.
The first forward_contrastive to use the encoder and projecter part and the forward while the forward for inference is
the encoder then a linear classifier.
The second is freeze_projection to freze weight learned on the encoder
'''

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
import torch
from .non_local_dot_product import NONLocalBlock1D, NONLocalBlock2D

class feature_fusion(nn.Module):
    def __init__(self):
        super(feature_fusion, self).__init__()

        #    self.conv1 = ResNet_Block(128, 128, False)
        #    self.conv2 = ResNet_Block(128, 128, False)

        self.conv = nn.Sequential(
            nn.Conv2d(128, 128, (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(128, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward1D(self, x):
        x_init = x.view(x.size(0), x.size(1),-1)
        out = self.conv1(x_init)
        att = F.softmax(out, dim=-1)
        out = x_init * att
        out = torch.sum(out, dim=-1, keepdim=True)
        out = torch.squeeze(out)
        return out

    def forward2D(self, x):
        out = self.conv(x)
        att = F.softmax(out, dim=-1)
        out = x * att
        # out = torch.sum(out, dim=-1, keepdim=True)

        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = F.normalize(out, dim=1)

        return out

    def forward(self, x):

        return self.forward1D(x)


class NonLocalBlock(nn.Module):
    def __init__(self, channel):
        super(NonLocalBlock, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = x.size()
        # [N, C/2, H * W]
        x_phi = self.conv_phi(x).view(b, c, -1)
        # [N, H * W, C/2]
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        # [N, H * W, H * W]
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        # [N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0,2,1).contiguous().view(b,self.inter_channel, h, w)
        # [N, C, H , W]
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x
        return out

class STN(nn.Module):
    def __init__(self, ):
        super(STN, self).__init__()
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=7),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
                nn.Conv2d(8, 10, kernel_size=5),
                nn.AdaptiveAvgPool2d((3,3)),
                nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)
        return x


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetContrastive(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, contrastive_dimension=128):
    #def __init__(self, block, num_blocks, num_classes=10, contrastive_dimension=64):
        super(ResNetContrastive, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.contrastive_hidden_layer = nn.Linear(64, contrastive_dimension)
        self.contrastive_output_layer = nn.Linear(contrastive_dimension, contrastive_dimension)

        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

        self.tsne_show=[]
        self.Flag = "Test"

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def freeze_projection(self):
        self.conv1.requires_grad_(False)
        self.bn1.requires_grad_(False)
        self.layer1.requires_grad_(False)
        self.layer2.requires_grad_(False)
        self.layer3.requires_grad_(False)

    def _forward_impl_encoder(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = F.normalize(out, dim=1)
        self.tsne_show = out

        return out

    def forward_constrative(self, x):
        # Implement from the encoder E to the projection network P
        x = self._forward_impl_encoder(x)
        #self.tsne_show=x

        x = self.contrastive_hidden_layer(x)
        x = F.relu(x)
        x = self.contrastive_output_layer(x)

        # Normalize to unit hypersphere
        x = F.normalize(x, dim=1)

        return x

    def SetFlag(self, Flag):
        self.Flag = Flag

    def forward(self, x, Flag="Train"):
    #def forward(self, x, Flag="Test"):

        # Implement from the encoder to the decoder network
        #x = self._forward_impl_encoder(x)
        #return self.linear(x)
        #Flag = self.Flag
        if Flag=="Train":

            x = self._forward_impl_encoder(x)

            x_classificatioin = self.linear(x)

            x = self.contrastive_hidden_layer(x)
            x = F.relu(x)
            x = self.contrastive_output_layer(x)

            # Normalize to unit hypersphere
            x_contrasive = F.normalize(x, dim=1)

            return x_contrasive, x_classificatioin

        if Flag=="Test":
           x = self._forward_impl_encoder(x)
           return self.linear(x)

    def forward_constrative_linear(self, x):
        # Implement from the encoder E to the projection network P
        x = self._forward_impl_encoder(x)

        x_classificatioin = self.linear(x)

        x = self.contrastive_hidden_layer(x)
        x = F.relu(x)
        x = self.contrastive_output_layer(x)

        # Normalize to unit hypersphere
        x_contrasive = F.normalize(x, dim=1)

        return x_contrasive, x_classificatioin

class MsNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
    #def __init__(self, block, num_blocks, num_classes=10, contrastive_dimension=64):
        super(MsNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.in_planes = 16
        self.layer1_1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer1_2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer1_3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.NonlocalBlock = NONLocalBlock2D(3, sub_sample=False, bn_layer=True)

        self.conv1_a = nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_a = nn.BatchNorm2d(16)
        self.conv1_b = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_b = nn.BatchNorm2d(1)

        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)

        self.in_planes = 16
        self.layer2_1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2_2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer2_3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.conv2_a = nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2_a = nn.BatchNorm2d(16)
        self.conv2_b = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2_b = nn.BatchNorm2d(1)

        self.conv3 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(16)

        self.in_planes = 16
        self.layer3_1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer3_2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3_3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.conv3_a = nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3_a = nn.BatchNorm2d(16)
        self.conv3_b = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3_b = nn.BatchNorm2d(1)

        self.attentionFC1 = nn.Linear(64 * 3, 64 * 3)
        self.attentionFC2 = nn.Linear(64 * 3, 64 * 3)

        self.fc_class = nn.Linear(64 * 3, num_classes)

        self.apply(_weights_init)

        #self.tsne_show=[]


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def freeze_projection(self):
        self.conv1.requires_grad_(False)
        self.bn1.requires_grad_(False)
        self.layer1.requires_grad_(False)
        self.layer2.requires_grad_(False)
        self.layer3.requires_grad_(False)

    def _forward_impl_encoder(self, Global, Local1, Local2):

        out1 = F.relu(self.bn1(self.conv1(Global)))
        out1 = self.layer1_1(out1)
        out1 = self.layer1_2(out1)
        out1 = self.layer1_3(out1)

        out2 = F.relu(self.bn1(self.conv2(Local1)))
        out2 = self.layer2_1(out2)
        out2 = self.layer2_2(out2)
        out2 = self.layer2_3(out2)

        out3 = F.relu(self.bn1(self.conv3(Local2)))
        out3 = self.layer3_1(out3)
        out3 = self.layer3_2(out3)
        out3 = self.layer3_3(out3)

        # ATTENTION LAYER
        Attention1 = self.conv1_a(out1)
        Attention1 = self.bn1_a(Attention1)
        Attention1 = self.relu(Attention1)
        Attention1 = self.conv1_b(Attention1)
        Attention1 = self.bn1_b(Attention1)

        Attention2 = self.conv2_a(out2)
        Attention2 = self.bn2_a(Attention2)
        Attention2 = self.relu(Attention2)
        Attention2 = self.conv2_b(Attention2)
        Attention2 = self.bn2_b(Attention2)

        Attention3 = self.conv3_a(out3)
        Attention3 = self.bn3_a(Attention3)
        Attention3 = self.relu(Attention3)
        Attention3 = self.conv3_b(Attention3)
        Attention3 = self.bn3_b(Attention3)

        AttentionConcat = torch.concat([Attention1, Attention2, Attention3],dim = 1)
        AttentionConcat = F.softmax(AttentionConcat, dim=1)

        out = F.avg_pool2d(Attention1, Attention1.size()[3])
        out = out.view(out.size(0), -1)
        out = F.normalize(out, dim=1)

        #self.tsne_show = out

        return out

    def _forward_impl_encoder2(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        # self.tsne_show = out

        return out

    def SetFlag(self, Flag):
        self.Flag = Flag

    def MSforward1(self, x ):
        # Centre Crop local patch 128 x 128
        x_local = x[:, :, 48:(48 + 128), 48:(48 + 128)]
        x_local = nn.functional.upsample_bilinear(x_local, (x.size()[2], x.size()[3]))

        # Centre Crop local patch 64 x 64
        x_local2 = x[:, :, 80:(80 + 64), 80:(80 + 64)]
        x_local2 = nn.functional.upsample_bilinear(x_local2, (x.size()[2], x.size()[3]))

        Global = x
        Local1 = x_local
        Local2 = x_local2

        out1 = F.relu(self.bn1(self.conv1(Global)))
        out1 = self.layer1_1(out1)
        out1 = self.layer1_2(out1)
        out1 = self.layer1_3(out1)

        out2 = F.relu(self.bn2(self.conv2(Local1)))
        out2 = self.layer2_1(out2)
        out2 = self.layer2_2(out2)
        out2 = self.layer2_3(out2)

        out3 = F.relu(self.bn3(self.conv3(Local2)))
        out3 = self.layer3_1(out3)
        out3 = self.layer3_2(out3)
        out3 = self.layer3_3(out3)

        # ATTENTION LAYER
        Attention1 = self.conv1_a(out1)
        Attention1 = self.bn1_a(Attention1)
        Attention1 = self.relu(Attention1)
        Attention1 = self.conv1_b(Attention1)
        Attention1 = self.bn1_b(Attention1)

        Attention2 = self.conv2_a(out2)
        Attention2 = self.bn2_a(Attention2)
        Attention2 = self.relu(Attention2)
        Attention2 = self.conv2_b(Attention2)
        Attention2 = self.bn2_b(Attention2)

        Attention3 = self.conv3_a(out3)
        Attention3 = self.bn3_a(Attention3)
        Attention3 = self.relu(Attention3)
        Attention3 = self.conv3_b(Attention3)
        Attention3 = self.bn3_b(Attention3)

        AttentionConcat = torch.concat([Attention1, Attention2, Attention3], dim=1)
        AttentionConcat = F.softmax(AttentionConcat, dim=1)

        SplitAttention1 = AttentionConcat[:, 0, :, :]
        SplitAttention2 = AttentionConcat[:, 1, :, :]
        SplitAttention3 = AttentionConcat[:, 2, :, :]

        SplitAttention1 = torch.unsqueeze(SplitAttention1, dim=1)
        SplitAttention2 = torch.unsqueeze(SplitAttention2, dim=1)
        SplitAttention3 = torch.unsqueeze(SplitAttention3, dim=1)

        out1 = SplitAttention1 * out1
        out2 = SplitAttention2 * out2
        out3 = SplitAttention3 * out3

        out1 = F.avg_pool2d(out1, out1.size()[3])
        out2 = F.avg_pool2d(out2, out2.size()[3])
        out3 = F.avg_pool2d(out3, out3.size()[3])

        out1 = out1.view(out1.size(0), -1)
        out1 = F.normalize(out1, dim=1)

        out2 = out2.view(out2.size(0), -1)
        out2 = F.normalize(out2, dim=1)

        out3 = out3.view(out3.size(0), -1)
        out3 = F.normalize(out3, dim=1)

        all_feature = torch.cat([out1, out2, out3], dim=1)

        out = self.fc_class(all_feature)

    def MSforwardSC(self, x):
        # Centre Crop local patch 128 x 128
        x_local = x[:, :, 48:(48 + 128), 48:(48 + 128)]
        x_local = nn.functional.upsample_bilinear(x_local, (x.size()[2], x.size()[3]))

        # Centre Crop local patch 64 x 64
        x_local2 = x[:, :, 80:(80 + 64), 80:(80 + 64)]
        x_local2 = nn.functional.upsample_bilinear(x_local2, (x.size()[2], x.size()[3]))

        Global = x
        Local1 = x_local
        Local2 = x_local2

        out1 = F.relu(self.bn1(self.conv1(Global)))
        out1 = self.layer1_1(out1)
        out1 = self.layer1_2(out1)
        out1 = self.layer1_3(out1)

        out2 = F.relu(self.bn2(self.conv2(Local1)))
        out2 = self.layer2_1(out2)
        out2 = self.layer2_2(out2)
        out2 = self.layer2_3(out2)

        out3 = F.relu(self.bn3(self.conv3(Local2)))
        out3 = self.layer3_1(out3)
        out3 = self.layer3_2(out3)
        out3 = self.layer3_3(out3)

        # Channal Attention LAYER
        Channal_out1 = F.avg_pool2d(out1, out1.size()[3])
        Channal_out2 = F.avg_pool2d(out2, out2.size()[3])
        Channal_out3 = F.avg_pool2d(out3, out3.size()[3])

        # Channal Block
        AllChannalAttention = torch.cat([Channal_out1, Channal_out2, Channal_out3], dim=1)
        #AllChannalAttention = torch.squeeze(AllChannalAttention)
        AllChannalAttention =  AllChannalAttention[:,:,0,0]

        AllChannalAttention_1 = self.attentionFC1(AllChannalAttention)
        AllChannalAttention_1 = self.relu(AllChannalAttention_1)
        AllChannalAttention_2 = self.attentionFC2(AllChannalAttention_1)

        #print(AllChannalAttention_2.size())
        AllChannalAttention_2 = F.softmax(AllChannalAttention_2, dim=1)

        SplitCAttention1 = AllChannalAttention_2[:, :64]
        SplitCAttention2 = AllChannalAttention_2[:, 64:128]
        SplitCAttention3 = AllChannalAttention_2[:, 128:196]

        SplitCAttention1 = torch.unsqueeze(SplitCAttention1, dim=-1)
        SplitCAttention1 = torch.unsqueeze(SplitCAttention1, dim=-1)

        SplitCAttention2 = torch.unsqueeze(SplitCAttention2, dim=-1)
        SplitCAttention2 = torch.unsqueeze(SplitCAttention2, dim=-1)

        SplitCAttention3 = torch.unsqueeze(SplitCAttention3, dim=-1)
        SplitCAttention3 = torch.unsqueeze(SplitCAttention3, dim=-1)

        out1 = out1 * SplitCAttention1
        out2 = out2 * SplitCAttention2
        out3 = out3 * SplitCAttention3

        # Spatial ATTENTION LAYER
        Attention1 = self.conv1_a(out1)
        Attention1 = self.bn1_a(Attention1)
        Attention1 = self.relu(Attention1)
        Attention1 = self.conv1_b(Attention1)
        Attention1 = self.bn1_b(Attention1)

        Attention2 = self.conv2_a(out2)
        Attention2 = self.bn2_a(Attention2)
        Attention2 = self.relu(Attention2)
        Attention2 = self.conv2_b(Attention2)
        Attention2 = self.bn2_b(Attention2)

        Attention3 = self.conv3_a(out3)
        Attention3 = self.bn3_a(Attention3)
        Attention3 = self.relu(Attention3)
        Attention3 = self.conv3_b(Attention3)
        Attention3 = self.bn3_b(Attention3)

        AttentionConcat = torch.concat([Attention1, Attention2, Attention3], dim=1)
        AttentionConcat = F.softmax(AttentionConcat, dim=1)

        SplitAttention1 = AttentionConcat[:, 0, :, :]
        SplitAttention2 = AttentionConcat[:, 1, :, :]
        SplitAttention3 = AttentionConcat[:, 2, :, :]

        SplitAttention1 = torch.unsqueeze(SplitAttention1, dim=1)
        SplitAttention2 = torch.unsqueeze(SplitAttention2, dim=1)
        SplitAttention3 = torch.unsqueeze(SplitAttention3, dim=1)

        # Spatial Attention
        out1 = SplitAttention1 * out1
        out2 = SplitAttention2 * out2
        out3 = SplitAttention3 * out3

        out1 = F.avg_pool2d(out1, out1.size()[3])
        out2 = F.avg_pool2d(out2, out2.size()[3])
        out3 = F.avg_pool2d(out3, out3.size()[3])

        out1 = out1.view(out1.size(0), -1)
        out1 = F.normalize(out1, dim=1)

        out2 = out2.view(out2.size(0), -1)
        out2 = F.normalize(out2, dim=1)

        out3 = out3.view(out3.size(0), -1)
        out3 = F.normalize(out3, dim=1)

        all_feature = torch.cat([out1, out2, out3], dim=1)

        out = self.fc_class(all_feature)

        return out

    def MSforwardSC_Risual(self, x):
        # Centre Crop local patch 128 x 128
        x_local = x[:, :, 48:(48 + 128), 48:(48 + 128)]
        x_local = nn.functional.upsample_bilinear(x_local, (x.size()[2], x.size()[3]))

        # Centre Crop local patch 64 x 64
        x_local2 = x[:, :, 80:(80 + 64), 80:(80 + 64)]
        x_local2 = nn.functional.upsample_bilinear(x_local2, (x.size()[2], x.size()[3]))

        Global = x
        Local1 = x_local
        Local2 = x_local2

        out1 = F.relu(self.bn1(self.conv1(Global)))
        out1 = self.layer1_1(out1)
        out1 = self.layer1_2(out1)
        out1 = self.layer1_3(out1)

        out2 = F.relu(self.bn2(self.conv2(Local1)))
        out2 = self.layer2_1(out2)
        out2 = self.layer2_2(out2)
        out2 = self.layer2_3(out2)

        out3 = F.relu(self.bn3(self.conv3(Local2)))
        out3 = self.layer3_1(out3)
        out3 = self.layer3_2(out3)
        out3 = self.layer3_3(out3)

        # Channal Attention LAYER
        Channal_out1 = F.avg_pool2d(out1, out1.size()[3])
        Channal_out2 = F.avg_pool2d(out2, out2.size()[3])
        Channal_out3 = F.avg_pool2d(out3, out3.size()[3])

        # Channal Block
        AllChannalAttention = torch.cat([Channal_out1, Channal_out2, Channal_out3], dim=1)
        AllChannalAttention =  AllChannalAttention[:,:,0,0]

        AllChannalAttention_1 = self.attentionFC1(AllChannalAttention)
        AllChannalAttention_1 = self.relu(AllChannalAttention_1)
        AllChannalAttention_2 = self.attentionFC2(AllChannalAttention_1)
        AllChannalAttention_2 = F.softmax(AllChannalAttention_2, dim=1)

        SplitCAttention1 = AllChannalAttention_2[:, :64]
        SplitCAttention2 = AllChannalAttention_2[:, 64:128]
        SplitCAttention3 = AllChannalAttention_2[:, 128:196]

        SplitCAttention1 = torch.unsqueeze(SplitCAttention1, dim=-1)
        SplitCAttention1 = torch.unsqueeze(SplitCAttention1, dim=-1)

        SplitCAttention2 = torch.unsqueeze(SplitCAttention2, dim=-1)
        SplitCAttention2 = torch.unsqueeze(SplitCAttention2, dim=-1)

        SplitCAttention3 = torch.unsqueeze(SplitCAttention3, dim=-1)
        SplitCAttention3 = torch.unsqueeze(SplitCAttention3, dim=-1)

        out1 = out1 + out1 * SplitCAttention1
        out2 = out2 + out2 * SplitCAttention2
        out3 = out3 + out3 * SplitCAttention3

        # Spatial ATTENTION LAYER
        Attention1 = self.conv1_a(out1)
        Attention1 = self.bn1_a(Attention1)
        Attention1 = self.relu(Attention1)
        Attention1 = self.conv1_b(Attention1)
        Attention1 = self.bn1_b(Attention1)

        Attention2 = self.conv2_a(out2)
        Attention2 = self.bn2_a(Attention2)
        Attention2 = self.relu(Attention2)
        Attention2 = self.conv2_b(Attention2)
        Attention2 = self.bn2_b(Attention2)

        Attention3 = self.conv3_a(out3)
        Attention3 = self.bn3_a(Attention3)
        Attention3 = self.relu(Attention3)
        Attention3 = self.conv3_b(Attention3)
        Attention3 = self.bn3_b(Attention3)

        AttentionConcat = torch.concat([Attention1, Attention2, Attention3], dim=1)

        AttentionConcat = F.softmax(AttentionConcat, dim=1)

        SplitAttention1 = AttentionConcat[:, 0, :, :]
        SplitAttention2 = AttentionConcat[:, 1, :, :]
        SplitAttention3 = AttentionConcat[:, 2, :, :]

        SplitAttention1 = torch.unsqueeze(SplitAttention1, dim=1)
        SplitAttention2 = torch.unsqueeze(SplitAttention2, dim=1)
        SplitAttention3 = torch.unsqueeze(SplitAttention3, dim=1)

        # Spatial Attention
        out1 = out1 + SplitAttention1 * out1
        out2 = out2 + SplitAttention2 * out2
        out3 = out3 + SplitAttention3 * out3

        out1 = F.avg_pool2d(out1, out1.size()[3])
        out2 = F.avg_pool2d(out2, out2.size()[3])
        out3 = F.avg_pool2d(out3, out3.size()[3])

        out1 = out1.view(out1.size(0), -1)
        out1 = F.normalize(out1, dim=1)

        out2 = out2.view(out2.size(0), -1)
        out2 = F.normalize(out2, dim=1)

        out3 = out3.view(out3.size(0), -1)
        out3 = F.normalize(out3, dim=1)

        all_feature = torch.cat([out1, out2, out3], dim=1)

        out = self.fc_class(all_feature)

        return out

    def forward(self,  x, feat):
        #out = self.MSforwardSC(x)
        out = self.MSforwardSC_Risual(x)
        return out

class ResNetBaseLine(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
    #def __init__(self, block, num_blocks, num_classes=10, contrastive_dimension=64):
        super(ResNetBaseLine, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.linear = nn.Linear(64, num_classes)
        self.apply(_weights_init)

        #self.tsne_show=[]


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def freeze_projection(self):
        self.conv1.requires_grad_(False)
        self.bn1.requires_grad_(False)
        self.layer1.requires_grad_(False)
        self.layer2.requires_grad_(False)
        self.layer3.requires_grad_(False)

    def _forward_impl_encoder(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = F.normalize(out, dim=1)

        #self.tsne_show = out

        return out

    def _forward_impl_encoder2(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        # self.tsne_show = out

        return out

    def SetFlag(self, Flag):
        self.Flag = Flag

    def forward(self, x):
        #x = self._forward_impl_encoder(x)
        x = self._forward_impl_encoder(x)

        #return self.linear(x)
        return x

class ResLayer(nn.Module):
    def __init__(self, linear_size):
        super(ResLayer, self).__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout()
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.dropout1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out = x + y

        return out

class TangNet(nn.Module):
    def __init__(self, ip_loc_dim, feats_dim, loc_dim, num_classes, use_loc):
        super(TangNet, self).__init__()
        self.use_loc  = use_loc
        self.fc_loc   = nn.Linear(ip_loc_dim, loc_dim)
        if self.use_loc:
            self.fc_class = nn.Linear(feats_dim+loc_dim, num_classes)
        else:
            self.fc_class = nn.Linear(feats_dim, num_classes)

    def forward(self, loc, net_feat):
        if self.use_loc:
            x = torch.sigmoid(self.fc_loc(loc))
            x = self.fc_class(torch.cat((x, net_feat), 1))
        else:
            x = self.fc_class(net_feat)
        return F.log_softmax(x, dim=1)

class FCNet(nn.Module):
    def __init__(self, num_inputs, num_classes, num_filts, num_users=1):
        super(FCNet, self).__init__()
        self.inc_bias = False
        self.class_emb = nn.Linear(num_filts, num_classes, bias=self.inc_bias)
        self.user_emb = nn.Linear(num_filts, num_users, bias=self.inc_bias)

        self.feats = nn.Sequential(nn.Linear(num_inputs, num_filts),
                                    nn.ReLU(inplace=True),
                                    ResLayer(num_filts),
                                    ResLayer(num_filts),
                                    ResLayer(num_filts),
                                    ResLayer(num_filts))

    def forward(self, x, class_of_interest=None, return_feats=False):
        loc_emb = self.feats(x)

        # if return_feats:
        return loc_emb

        # if class_of_interest is None:
        #     class_pred = self.class_emb(loc_emb)
        # else:
        #     class_pred = self.eval_single_class(loc_emb, class_of_interest)

        #return torch.sigmoid(class_pred)

class ResNetGeo(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, contrastive_dimension=128):
    #def __init__(self, block, num_blocks, num_classes=10, contrastive_dimension=64):
        super(ResNetGeo, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        #self.GeoNet = FCNet(8, num_classes, 16)

        self.GeoNet = FCNet(2, num_classes, 64)

        #self.linear1 = nn.Linear(64, 128)
        #self.linear2 = nn.Linear(256, 128)

        #self.fc_class = nn.Linear(64 + 16, num_classes)
        self.fc_class = nn.Linear(64, num_classes)

        self.apply(_weights_init)

        self.tsne_show=[]
        self.Flag = "Test"

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def _forward_impl_encoder(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)

        return out

    def forward_geo2(self, x, feat):
        # Implement from the encoder E to the projection network P
        x = self._forward_impl_encoder(x)
        GoEmbedding = self.GeoNet(feat)
        GoEmbedding = F.normalize(GoEmbedding, dim=1)

        # Concat
        #FeatureConcat = torch.cat([x, GoEmbedding], dim = 1)

        # Plus
        FeaturePlus = x + GoEmbedding

        #Feature1 = self.linear1(x)
        #FeatureConcat = torch.cat([x, GoEmbedding], dim=1)

        #Feature2 = self.linear2(Feature1)
        #out = self.fc_class(Feature2)

        out = self.fc_class(FeaturePlus)
        #out = self.fc_class(x)
        out = F.normalize(out, dim=1)

        return out

    def forward_geo(self, x, feat):
        # Implement from the encoder E to the projection network P
        x = self._forward_impl_encoder(x)
        GoEmbedding = self.GeoNet(feat)
        GoEmbedding = F.normalize(GoEmbedding, dim=1)

        # Concat
        FeatureConcat = torch.cat([x, GoEmbedding], dim = 1)

        #Feature1 = self.linear1(x)
        #FeatureConcat = torch.cat([x, GoEmbedding], dim=1)

        #Feature2 = self.linear2(Feature1)
        #out = self.fc_class(Feature2)

        out = self.fc_class(FeatureConcat)
        #out = self.fc_class(x)
        out = F.normalize(out, dim=1)

        return out

    def SetFlag(self, Flag):
        self.Flag = Flag

    def forward(self, x, feat):
        return self.forward_geo2(x, feat)


class ResNetLocalGlobal(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, contrastive_dimension=128):
    #def __init__(self, block, num_blocks, num_classes=10, contrastive_dimension=64):
        super(ResNetLocalGlobal, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        #self.GeoNet = FCNet(8, num_classes, 16)

        #self.linear1 = nn.Linear(64, 128)
        #self.linear2 = nn.Linear(256, 128)

        #self.fc_class = nn.Linear(64 + 16, num_classes)
        self.fc_class = nn.Linear(64 * 3, num_classes)

        self.apply(_weights_init)

        self.tsne_show=[]
        self.Flag = "Test"

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def _forward_impl_encoder(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)

        return out

    def forward_local_global(self, x, feat):
        # Centre Crop local patch 128 x 128
        x_local = x[:,:,48:(48+128),48:(48+128)]

        # Implement from the encoder E to the projection network P
        x_global_feature = self._forward_impl_encoder(x)

        x_local_feature = self._forward_impl_encoder(x_local)

        # Concat
        FeatureConcat = torch.cat([x_global_feature, x_local_feature], dim = 1)

        #Feature1 = self.linear1(x)
        #FeatureConcat = torch.cat([x, GoEmbedding], dim=1)

        #Feature2 = self.linear2(Feature1)
        #out = self.fc_class(Feature2)

        #out = self.fc_class(FeatureConcat)
        out = self.fc_class(FeatureConcat)
        out = F.normalize(out, dim=1)

        return out

    def forward_local_global2(self, x, feat):
        # Centre Crop local patch 128 x 128
        x_local = x[:, :, 48:(48 + 128), 48:(48 + 128)]

        # Implement from the encoder E to the projection network P
        x_global_feature = self._forward_impl_encoder(x)

        x_local_feature = self._forward_impl_encoder(x_local)

        # Concat
        FeatureConcat = torch.cat([x_global_feature, x_local_feature], dim=1)

        # Feature1 = self.linear1(x)
        # FeatureConcat = torch.cat([x, GoEmbedding], dim=1)

        # Feature2 = self.linear2(Feature1)
        # out = self.fc_class(Feature2)

        # out = self.fc_class(FeatureConcat)
        out = self.fc_class(FeatureConcat)
        out = F.normalize(out, dim=1)

        return out

    def SetFlag(self, Flag):
        self.Flag = Flag

    def forward(self, x, feat):
        return self.forward_local_global(x, feat)

class ResNetGlobalEdge(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, contrastive_dimension=128):
    #def __init__(self, block, num_blocks, num_classes=10, contrastive_dimension=64):
        super(ResNetGlobalEdge, self).__init__()

        self.Network1 = ResNetBaseLine(BasicBlock, [5, 5, 5], num_classes)
        #self.Network2 = ResNetBaseLine(BasicBlock, [5, 5, 5], num_classes)
        #self.Network3 = ResNetBaseLine(BasicBlock, [5, 5, 5], num_classes)

        #self.STN_Network = STN()
        #self.feature_fusion_net = feature_fusion()
        #self.NonLocalFeature = NONLocalBlock2D(3, sub_sample=False, bn_layer=True)
        #self.Network2

        #self.GeoNet = FCNet(8, num_classes, 16)

        #self.linear1 = nn.Linear(64, 128)
        #self.linear2 = nn.Linear(256, 128)

        #self.fc_class = nn.Linear(64 + 16, num_classes)
        self.fc_class = nn.Linear(64, num_classes)

        self.apply(_weights_init)

        self.tsne_show=[]
        self.Flag = "Test"

    def forward_global_Edge10(self, x, feat):
        # Global + local + late Fusion

        # Centre Crop local patch 128 x 128
        x_local = x[:, :, 48:(48 + 128), 48:(48 + 128)]
        x_local = nn.functional.upsample_bilinear(x_local, (x.size()[2], x.size()[3]))

        x_global_feature = self.Network1(x)

        x_local_feature = self.Network2(x_local)

        # Concat
        FeatureConcat = torch.cat([x_global_feature, x_local_feature], dim=1)

        # NonLocal Attention
        FeatureConcat = NONLocalBlock2D(FeatureConcat)

        # xfusion = self.feature_fusion_net(FeatureConcat)

        # out = F.avg_pool2d(xfusion, xfusion.size()[3])
        # out = xfusion.view(xfusion.size(0), -1)
        # out = F.normalize(out, dim=1)

        # Feature1 = self.linear1(x)
        # FeatureConcat = torch.cat([x, GoEmbedding], dim=1)

        # Feature2 = self.linear2(Feature1)
        # out = self.fc_class(Feature2)
        out = F.avg_pool2d(FeatureConcat, FeatureConcat.size()[3])
        out = out.view(out.size(0), -1)
        out = F.normalize(out, dim=1)

        # out = self.fc_class(FeatureConcat)
        out = self.fc_class(out)

        out = F.normalize(out, dim=1)

        return out

    def forward_global_Edge9(self, x, feat):
        # Global + local + late Fusion

        # Centre Crop local patch 128 x 128
        x_local = x[:, :, 48:(48 + 128), 48:(48 + 128)]
        x_local = nn.functional.upsample_bilinear(x_local, (x.size()[2], x.size()[3]))

        x_global_feature = self.Network1(x)

        x_local_feature = self.Network2(x_local)

        # Concat
        FeatureConcat = torch.cat([x_global_feature, x_local_feature], dim=1)

        # xfusion = self.feature_fusion_net(FeatureConcat)

        # out = F.avg_pool2d(xfusion, xfusion.size()[3])
        # out = xfusion.view(xfusion.size(0), -1)
        # out = F.normalize(out, dim=1)

        # Feature1 = self.linear1(x)
        # FeatureConcat = torch.cat([x, GoEmbedding], dim=1)

        # Feature2 = self.linear2(Feature1)
        # out = self.fc_class(Feature2)
        out = F.avg_pool2d(FeatureConcat, FeatureConcat.size()[3])
        out = out.view(out.size(0), -1)
        out = F.normalize(out, dim=1)

        # out = self.fc_class(FeatureConcat)
        out = self.fc_class(out)

        out = F.normalize(out, dim=1)

        return out


    def forward_global_Edge8(self, x, feat):
        # Global + local upsample

        # Centre Crop local patch 128 x 128
        x_local = x[:, :, 48:(48 + 128), 48:(48 + 128)]
        x_local = nn.functional.upsample_bilinear(x_local, (x.size()[2], x.size()[3]))

        # Centre Crop local patch 64 x 64
        x_local2 = x[:, :, 80:(80 + 64), 80:(80 + 64)]
        x_local2 = nn.functional.upsample_bilinear(x_local2, (x.size()[2], x.size()[3]))

        # Implement from the encoder E to the projection network P

        x_global_feature = self.Network1(x)

        x_local_feature = self.Network2(x_local)

        x_local_feature3 = self.Network3(x_local2)

        # Concat
        FeatureConcat = torch.cat([x_global_feature, x_local_feature, x_local_feature3], dim=1)
        # xfusion = self.feature_fusion_net(FeatureConcat)

        # out = F.avg_pool2d(xfusion, xfusion.size()[3])
        # out = xfusion.view(xfusion.size(0), -1)
        # out = F.normalize(out, dim=1)

        # Feature1 = self.linear1(x)
        # FeatureConcat = torch.cat([x, GoEmbedding], dim=1)

        # Feature2 = self.linear2(Feature1)
        # out = self.fc_class(Feature2)

        # out = self.fc_class(FeatureConcat)
        out = self.fc_class(FeatureConcat)
        out = F.normalize(out, dim=1)

        return out

    def forward_global_Edge7(self, x, feat):
        # Global + local upsample

        # Centre Crop local patch 128 x 128
        x_local = x[:, :, 48:(48 + 128), 48:(48 + 128)]
        x_local = nn.functional.upsample_bilinear(x_local, (x.size()[2], x.size()[3]))

        # Implement from the encoder E to the projection network P

        x_global_feature = self.Network1(x)

        x_local_feature = self.Network2(x_local)

        # STNfeature = self.STN_Network(x)
        # x_edge_rediual = x - STNfeature
        # x_edge_feature = self.Network2(x_edge_rediual)

        # Concat
        FeatureConcat = torch.cat([x_global_feature, x_local_feature], dim=1)
        #xfusion = self.feature_fusion_net(FeatureConcat)

        # out = F.avg_pool2d(xfusion, xfusion.size()[3])
        # out = xfusion.view(xfusion.size(0), -1)
        # out = F.normalize(out, dim=1)

        # Feature1 = self.linear1(x)
        # FeatureConcat = torch.cat([x, GoEmbedding], dim=1)

        # Feature2 = self.linear2(Feature1)
        # out = self.fc_class(Feature2)

        # out = self.fc_class(FeatureConcat)
        out = self.fc_class(FeatureConcat)
        out = F.normalize(out, dim=1)

        return out

    def forward_global_Edge6(self, x, feat):
        # Global + local non_upsample   3 scale 128x 128 64x64

        # Centre Crop local patch 128 x 128
        x_local = x[:, :, 48:(48 + 128), 48:(48 + 128)]

        # Implement from the encoder E to the projection network P
        x_local = nn.functional.upsample_bilinear(x_local, (x.size()[2], x.size()[3]))

        x_global_feature = self.Network1(x)

        x_local_feature = self.Network2(x_local)

        # STNfeature = self.STN_Network(x)
        # x_edge_rediual = x - STNfeature
        # x_edge_feature = self.Network2(x_edge_rediual)

        # Concat
        FeatureConcat = torch.cat([x_global_feature, x_local_feature], dim=1)
        xfusion = self.feature_fusion_net(FeatureConcat)

        # out = F.avg_pool2d(xfusion, xfusion.size()[3])
        # out = xfusion.view(xfusion.size(0), -1)
        # out = F.normalize(out, dim=1)

        # Feature1 = self.linear1(x)
        # FeatureConcat = torch.cat([x, GoEmbedding], dim=1)

        # Feature2 = self.linear2(Feature1)
        # out = self.fc_class(Feature2)

        # out = self.fc_class(FeatureConcat)
        out = self.fc_class(xfusion)
        out = F.normalize(out, dim=1)

        return out


    def forward_global_Edge5(self, x, feat):
        # Global + local non_upsample   3 scale 128x 128 64x64




        # Centre Crop local patch 128 x 128
        x_local = x[:, :, 48:(48 + 128), 48:(48 + 128)]

        # Implement from the encoder E to the projection network P
        x_local = nn.functional.upsample_bilinear(x_local, (x.size()[2],x.size()[3]))

        x_global_feature = self.Network1(x)

        x_local_feature = self.Network2(x_local)

        #STNfeature = self.STN_Network(x)
        #x_edge_rediual = x - STNfeature
        #x_edge_feature = self.Network2(x_edge_rediual)

        # Concat
        FeatureConcat = torch.cat([x_global_feature, x_local_feature], dim=1)
        xfusion = self.feature_fusion_net(FeatureConcat)

        #out = F.avg_pool2d(xfusion, xfusion.size()[3])
        #out = xfusion.view(xfusion.size(0), -1)
        #out = F.normalize(out, dim=1)

        # Feature1 = self.linear1(x)
        # FeatureConcat = torch.cat([x, GoEmbedding], dim=1)

        # Feature2 = self.linear2(Feature1)
        # out = self.fc_class(Feature2)

        # out = self.fc_class(FeatureConcat)
        out = self.fc_class(xfusion)
        out = F.normalize(out, dim=1)

        return out

    def forward_global_Edge4(self, x, feat):
        # Centre Crop local patch 128 x 128
        x_local = x[:, :, 48:(48 + 128), 48:(48 + 128)]

        # Implement from the encoder E to the projection network P
        x_local = nn.functional.upsample_bilinear(x_local, (x.size()[2],x.size()[3]))

        x_global_feature = self.Network1(x)

        x_local_feature = self.Network2(x_local)

        #STNfeature = self.STN_Network(x)
        #x_edge_rediual = x - STNfeature
        #x_edge_feature = self.Network2(x_edge_rediual)

        # Concat
        FeatureConcat = torch.cat([x_global_feature, x_local_feature], dim=1)
        xfusion = self.feature_fusion_net(FeatureConcat)

        #out = F.avg_pool2d(xfusion, xfusion.size()[3])
        #out = xfusion.view(xfusion.size(0), -1)
        #out = F.normalize(out, dim=1)

        # Feature1 = self.linear1(x)
        # FeatureConcat = torch.cat([x, GoEmbedding], dim=1)

        # Feature2 = self.linear2(Feature1)
        # out = self.fc_class(Feature2)

        # out = self.fc_class(FeatureConcat)
        out = self.fc_class(xfusion)
        out = F.normalize(out, dim=1)

        return out

    def forward_global_Edge3(self, x, feat):
        # Centre Crop local patch 128 x 128
        x_local = x[:, :, 48:(48 + 128), 48:(48 + 128)]

        # Implement from the encoder E to the projection network P
        x_local = nn.functional.upsample_bilinear(x_local, (x.size()[2],x.size()[3]))

        x_global_feature = self.Network1(x)

        x_local_feature = self.Network2(x_local)

        #STNfeature = self.STN_Network(x)
        #x_edge_rediual = x - STNfeature
        #x_edge_feature = self.Network2(x_edge_rediual)

        # Concat
        FeatureConcat = torch.cat([x_global_feature, x_local_feature], dim=1)
        xfusion = self.feature_fusion_net(FeatureConcat)

        #out = F.avg_pool2d(xfusion, xfusion.size()[3])
        #out = xfusion.view(xfusion.size(0), -1)
        #out = F.normalize(out, dim=1)

        # Feature1 = self.linear1(x)
        # FeatureConcat = torch.cat([x, GoEmbedding], dim=1)

        # Feature2 = self.linear2(Feature1)
        # out = self.fc_class(Feature2)

        # out = self.fc_class(FeatureConcat)
        out = self.fc_class(xfusion)
        out = F.normalize(out, dim=1)

        return out

    def forward_global_Edge2(self, x, feat):
        # Centre Crop local patch 128 x 128
        x_local = x[:,:,48:(48+128),48:(48+128)]

        # Implement from the encoder E to the projection network P
        x_global_feature = self.Network1(x)

        #x_local_feature = self.Network2(x_local)
        STNfeature = self.STN_Network(x)
        x_edge_rediual = x - STNfeature
        x_edge_feature = self.Network2(x_edge_rediual)

        # Concat
        FeatureConcat = torch.cat([x_global_feature, x_edge_feature], dim = 1)

        #Feature1 = self.linear1(x)
        #FeatureConcat = torch.cat([x, GoEmbedding], dim=1)

        #Feature2 = self.linear2(Feature1)
        #out = self.fc_class(Feature2)

        #out = self.fc_class(FeatureConcat)
        out = self.fc_class(FeatureConcat)
        out = F.normalize(out, dim=1)

        return out

    def forward_global_Edge(self, x, feat):
        # Centre Crop local patch 128 x 128
        x_local = x[:,:,48:(48+128),48:(48+128)]

        # Implement from the encoder E to the projection network P
        x_global_feature = self.Network1(x)

        x_local_feature = self.Network2(x_local)
        #x_edge_feature = self._forward_impl_encoder(x_local)

        # Concat
        FeatureConcat = torch.cat([x_global_feature, x_local_feature], dim = 1)

        #Feature1 = self.linear1(x)
        #FeatureConcat = torch.cat([x, GoEmbedding], dim=1)

        #Feature2 = self.linear2(Feature1)
        #out = self.fc_class(Feature2)

        #out = self.fc_class(FeatureConcat)
        out = self.fc_class(FeatureConcat)
        out = F.normalize(out, dim=1)

        return out

    def SetFlag(self, Flag):
        self.Flag = Flag

    def forward_baseline_global(self, x, feat):
        # Centre Crop local patch 128 x 128
        #x_local = x[:,:,48:(48+128),48:(48+128)]

        # Implement from the encoder E to the projection network P
        x_global_feature = self.Network1(x)

        #x_local_feature = self.Network2(x_local)
        #x_edge_feature = self._forward_impl_encoder(x_local)

        # Concat
        #FeatureConcat = torch.cat([x_global_feature, x_local_feature], dim = 1)

        #Feature1 = self.linear1(x)
        #FeatureConcat = torch.cat([x, GoEmbedding], dim=1)

        #Feature2 = self.linear2(Feature1)
        #out = self.fc_class(Feature2)

        #out = self.fc_class(FeatureConcat)
        out = self.fc_class(x_global_feature)
        out = F.normalize(out, dim=1)

        return out

    def forward_multiscale(self, x, feat):
        # Centre Crop local patch 128 x 128
        x_local1 = x[:, :, 48:(48 + 128),   48:(48+128)]
        x_local2 = x[:, :, 80:(80 + 64), 80:(80 + 64)]

        # Implement from the encoder E to the projection network P
        x_global_feature = self.Network1(x_local)




        #x_local_feature = self.Network2(x_local)
        #x_edge_feature = self._forward_impl_encoder(x_local)

        # Concat
        #FeatureConcat = torch.cat([x_global_feature, x_local_feature], dim = 1)

        #Feature1 = self.linear1(x)
        #FeatureConcat = torch.cat([x, GoEmbedding], dim=1)

        #Feature2 = self.linear2(Feature1)
        #out = self.fc_class(Feature2)

        #out = self.fc_class(FeatureConcat)
        out = self.fc_class(x_global_feature)
        out = F.normalize(out, dim=1)

        return out

    def forward_baseline_local(self, x, feat):
        # Centre Crop local patch 128 x 128
        x_local = x[:,:,48:(48+128),48:(48+128)]

        # Implement from the encoder E to the projection network P
        x_global_feature = self.Network1(x_local)

        #x_local_feature = self.Network2(x_local)
        #x_edge_feature = self._forward_impl_encoder(x_local)

        # Concat
        #FeatureConcat = torch.cat([x_global_feature, x_local_feature], dim = 1)

        #Feature1 = self.linear1(x)
        #FeatureConcat = torch.cat([x, GoEmbedding], dim=1)

        #Feature2 = self.linear2(Feature1)
        #out = self.fc_class(Feature2)

        #out = self.fc_class(FeatureConcat)
        out = self.fc_class(x_global_feature)
        out = F.normalize(out, dim=1)

        return out

    def SetFlag(self, Flag):
        self.Flag = Flag

    def forward(self, x, feat):
        return self.forward_baseline_local(x, feat)

class ResNetLocalGlobalGeo(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, contrastive_dimension=128):
    #def __init__(self, block, num_blocks, num_classes=10, contrastive_dimension=64):
        super(ResNetLocalGlobalGeo, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.GeoNet = FCNet(8, num_classes, 16)

        #self.linear1 = nn.Linear(64, 128)
        #self.linear2 = nn.Linear(256, 128)

        #self.fc_class = nn.Linear(64 + 16, num_classes)
        self.fc_class = nn.Linear(128 + 16, num_classes)

        self.apply(_weights_init)

        self.tsne_show=[]
        self.Flag = "Test"

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def _forward_impl_encoder(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)

        return out

    def forward_local_global(self, x, feat):
        # Centre Crop local patch 128 x 128
        x_local = x[:,:,48:(48+128),48:(48+128)]

        # Implement from the encoder E to the projection network P
        x_global_feature = self._forward_impl_encoder(x)

        x_local_feature = self._forward_impl_encoder(x_local)

        # Concat
        FeatureConcat = torch.cat([x_global_feature, x_local_feature], dim = 1)

        #Feature1 = self.linear1(x)
        #FeatureConcat = torch.cat([x, GoEmbedding], dim=1)

        #Feature2 = self.linear2(Feature1)
        #out = self.fc_class(Feature2)

        #out = self.fc_class(FeatureConcat)
        out = self.fc_class(FeatureConcat)
        out = F.normalize(out, dim=1)

        return out

    def forward_local_global_geo(self, x, feat):
        # Centre Crop local patch 128 x 128
        x_local = x[:,:,48:(48+128),48:(48+128)]

        # Implement from the encoder E to the projection network P
        x_global_feature = self._forward_impl_encoder(x)

        x_local_feature = self._forward_impl_encoder(x_local)

        GoEmbedding = self.GeoNet(feat)
        GoEmbedding = F.normalize(GoEmbedding, dim=1)

        # Concat
        FeatureConcat = torch.cat([x_global_feature, x_local_feature, GoEmbedding], dim = 1)

        #Feature1 = self.linear1(x)
        #FeatureConcat = torch.cat([x, GoEmbedding], dim=1)

        #Feature2 = self.linear2(Feature1)
        #out = self.fc_class(Feature2)

        #out = self.fc_class(FeatureConcat)
        out = self.fc_class(FeatureConcat)
        out = F.normalize(out, dim=1)

        return out


    def SetFlag(self, Flag):
        self.Flag = Flag

    def forward(self, x, feat):
        return self.forward_local_global_geo(x, feat)


def resnet20_contrastive(num_classes=10):
    return ResNetContrastive(BasicBlock, [3, 3, 3], num_classes)

def resnet32_contrastive(num_classes=10):
    return ResNetContrastive(BasicBlock, [5, 5, 5], num_classes)

def resnet32_geo(num_classes=10):
    return ResNetGeo(BasicBlock, [5, 5, 5], num_classes)

def resnet32_local_global(num_classes=10):
    return ResNetLocalGlobal(BasicBlock, [5, 5, 5], num_classes)

def resnet32_global_edge(num_classes=10):
    return ResNetGlobalEdge(BasicBlock, [5, 5, 5], num_classes)

def resnet32_local_global_geo(num_classes=10):
    return ResNetLocalGlobalGeo(BasicBlock, [5, 5, 5], num_classes)

def resnet32_ms(num_classes=10):
    return MsNet(BasicBlock, [5, 5, 5], num_classes)
def resnet56_ms(num_classes=10):
    return MsNet(BasicBlock, [9, 9, 9], num_classes)
def resnet110_ms(num_classes=10):
    return MsNet(BasicBlock, [18, 18, 18], num_classes)

def resnet44_contrastive(num_classes=10):
    return ResNetContrastive(BasicBlock, [7, 7, 7], num_classes)

def resnet56_contrastive(num_classes=10):
    return ResNetContrastive(BasicBlock, [9, 9, 9], num_classes)

def resnet110_contrastive(num_classes=10):
    return ResNetContrastive(BasicBlock, [18, 18, 18], num_classes)


def resnet1202_contrastive(num_classes=10):
    return ResNetContrastive(BasicBlock, [200, 200, 200], num_classes)

def Vgg16(num_classes=10):
    vgg16 = torchvision.models.vgg16(pretrained = False)
    print(vgg16)
    vgg16.classifier[6] = nn.Linear(4096, num_classes)
    return vgg16

def GoogleNet(num_classes=10):
    googlenet = torchvision.models.googlenet(pretrained = False)
    print(googlenet)
    googlenet.fc = nn.Linear(1024, num_classes)
    return googlenet

def DenseNet121(num_classes=10):
    densenet = torchvision.models.densenet121(pretrained = False)
    print(densenet)
    densenet.classifier = nn.Linear(1024, num_classes)
    return densenet

def get_resnet_contrastive(name='resnet20', num_classes=10):
    """
    Get back a resnet model tailored for cifar datasets and dimension which has been modified for a contrastive approach

    :param name: str
    :param num_classes: int
    :return: torch.nn.Module
    """
    name_to_resnet = {'resnet20': resnet20_contrastive, 'resnet32': resnet32_contrastive,
                      'resnet44': resnet44_contrastive, 'resnet56': resnet56_contrastive,
                      'resnet110': resnet110_contrastive, 'resnet1202': resnet1202_contrastive,
                      'vgg16':Vgg16 , 'GoogleNet':GoogleNet, 'DenseNet121':DenseNet121}

    if name in name_to_resnet:
        return name_to_resnet[name](num_classes)
    else:
        raise ValueError('Model name %s not found'.format(name))

def get_resnet_geo(name='resnet20', num_classes=10):
    """
    Get back a resnet model tailored for cifar datasets and dimension which has been modified for a contrastive approach

    :param name: str
    :param num_classes: int
    :return: torch.nn.Module
    """
    name_to_resnet = {'resnet20': resnet20_contrastive, 'resnet32': resnet32_geo,
                      'resnet44': resnet44_contrastive, 'resnet56': resnet56_contrastive,
                      'resnet110': resnet110_contrastive, 'resnet1202': resnet1202_contrastive,
                      'vgg16':Vgg16 , 'GoogleNet':GoogleNet, 'DenseNet121':DenseNet121}

    if name in name_to_resnet:
        return name_to_resnet[name](num_classes)
    else:
        raise ValueError('Model name %s not found'.format(name))

def get_resnet_local_Global(name='resnet20', num_classes=10):
    """
    Get back a resnet model tailored for cifar datasets and dimension which has been modified for a contrastive approach

    :param name: str
    :param num_classes: int
    :return: torch.nn.Module
    """
    name_to_resnet = {'resnet20': resnet20_contrastive, 'resnet32': resnet32_local_global,
                      'resnet44': resnet44_contrastive, 'resnet56': resnet56_contrastive,
                      'resnet110': resnet110_contrastive, 'resnet1202': resnet1202_contrastive,
                      'vgg16':Vgg16 , 'GoogleNet':GoogleNet, 'DenseNet121':DenseNet121}

    if name in name_to_resnet:
        return name_to_resnet[name](num_classes)
    else:
        raise ValueError('Model name %s not found'.format(name))

def get_resnet_Global_Edge(name='resnet20', num_classes=10):
    """
    Get back a resnet model tailored for cifar datasets and dimension which has been modified for a contrastive approach

    :param name: str
    :param num_classes: int
    :return: torch.nn.Module
    """
    name_to_resnet = {'resnet20': resnet20_contrastive, 'resnet32': resnet32_global_edge,
                      'resnet44': resnet44_contrastive, 'resnet56': resnet56_contrastive,
                      'resnet110': resnet110_contrastive, 'resnet1202': resnet1202_contrastive,
                      'vgg16':Vgg16 , 'GoogleNet':GoogleNet, 'DenseNet121':DenseNet121}

    if name in name_to_resnet:
        return name_to_resnet[name](num_classes)
    else:
        raise ValueError('Model name %s not found'.format(name))

def get_resnet_local_Global_geo(name='resnet20', num_classes=10):
    """
    Get back a resnet model tailored for cifar datasets and dimension which has been modified for a contrastive approach

    :param name: str
    :param num_classes: int
    :return: torch.nn.Module
    """
    name_to_resnet = {'resnet20': resnet20_contrastive, 'resnet32': resnet32_local_global_geo,
                      'resnet44': resnet44_contrastive, 'resnet56': resnet56_contrastive,
                      'resnet110': resnet110_contrastive, 'resnet1202': resnet1202_contrastive,
                      'vgg16':Vgg16 , 'GoogleNet':GoogleNet, 'DenseNet121':DenseNet121}

    if name in name_to_resnet:
        return name_to_resnet[name](num_classes)
    else:
        raise ValueError('Model name %s not found'.format(name))

def get_resnet_ms(name='resnet20', num_classes=10):
    """
    Get back a resnet model tailored for cifar datasets and dimension which has been modified for a contrastive approach

    :param name: str
    :param num_classes: int
    :return: torch.nn.Module
    """
    name_to_resnet = {'resnet20': resnet20_contrastive, 'resnet32': resnet32_ms,
                      'resnet44': resnet44_contrastive, 'resnet56': resnet56_ms,
                      'resnet110': resnet110_ms, 'resnet1202': resnet1202_contrastive,
                      'vgg16':Vgg16 , 'GoogleNet':GoogleNet, 'DenseNet121':DenseNet121}

    if name in name_to_resnet:
        return name_to_resnet[name](num_classes)
    else:
        raise ValueError('Model name %s not found'.format(name))
