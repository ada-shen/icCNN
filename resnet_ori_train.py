#!/usr/bin/env python

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from load_utils import load_state_dict_from_url
from cub_voc import CUB_VOC
from celeb import Celeb
import os
from tqdm import tqdm
import shutil
import numpy as np
from newPad2d import newPad2d

IS_TRAIN = 0        # 0/1
IS_MULTI = 0
LAYERS = '18'
DATANAME = 'bird'  #
NUM_CLASSES = 6 if IS_MULTI else 2
if DATANAME == 'celeb':
    NUM_CLASSES = 80
cub_file = '/data/sw/dataset/frac_dataset'
voc_file = '/data/sw/dataset/VOCdevkit/VOC2010/voc2010_crop'
celeb_file = '/home/user05/fjq/dataset/CelebA/'
log_path = '/data/fjq/iccnn/resnet/' # for model
save_path = '/data/fjq/iccnn/basic_fmap/resnet/'  # for get_feature
acc_path = '/data/fjq/iccnn/basic_fmap/resnet/acc/'

dataset = '%s_resnet_%s_ori' % (LAYERS, DATANAME)
log_path = log_path + dataset + '/'
pretrain_model = log_path + 'model_2000.pth'
BATCHSIZE = 16
LR = 0.000001
EPOCH = 200


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=0, groups=groups, bias=False, dilation=dilation)#new padding

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.pad2d = newPad2d(1) #nn.ReplicationPad2d(1)#new paddig

    def forward(self, x):
        identity = x
        out = self.pad2d(x) #new padding
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.pad2d(out) #new padding
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.pad2d = newPad2d(1) #nn.ReplicationPad2d(1)#new paddig

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.pad2d(out) #new padding
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=0,
                               bias=False)#new padding
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)#new padding
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.pad2d_1 = newPad2d(1)#new paddig
        self.pad2d_3 = newPad2d(3)#new paddig

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        # See note [TorchScript super()]
        x = self.pad2d_3(x) #new padding
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pad2d_1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        f_map = x.detach()
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x, f_map

def _resnet(arch, block, layers, n_cls, pretrained, progress, **kwargs):
    model = ResNet(block, layers, n_cls, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        pretrained_dict = {k: v for k, v in state_dict.items() if 'fc' not in k and 'layer4.1' not in k}#'fc' not in k and 'layer4.1' not in k and
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    else:
        device = torch.device("cuda")
        model = nn.DataParallel(model).to(device)
        model.load_state_dict(torch.load(pretrain_model))
    return model

def ResNet18(n_cls, pretrained=False, progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock, [2,2,2,2], n_cls, pretrained, progress, **kwargs)

def ResNet34(n_cls, pretrained=False, progress=True, **kwargs):
    return _resnet('resnet34', BasicBlock, [3,4,6,3], n_cls, pretrained, progress, **kwargs)

def ResNet50(n_cls, pretrained=False, progress=True, **kwargs):
    return _resnet('resnet50', Bottleneck, [3,4,6,3], n_cls, pretrained, progress, **kwargs)

def ResNet101(n_cls, pretrained=False, progress=True, **kwargs):
    return _resnet('resnet101', Bottleneck, [3,4,23,3], n_cls, pretrained, progress, **kwargs)

def ResNet152(n_cls, pretrained=False, progress=True, **kwargs):
    return _resnet('resnet152', Bottleneck, [3,8,36,3], n_cls, pretrained, progress, **kwargs)

def get_Data(is_train, dataset_name,batch_size):
    transform = transforms.Compose([
        transforms.RandomResizedCrop((224,224),scale=(0.5,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    val_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    voc_helen_name = ['bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'helen', 'voc_multi']
    ##cub dataset###
    label = None if is_train else 0
    if dataset_name == 'cub':
        trainset = CUB_VOC(cub_file, dataset_name, 'ori', train=True, transform=transform, is_frac=label)
        testset = CUB_VOC(cub_file, dataset_name, 'ori', train=False, transform=val_transform, is_frac=label)
    ###cropped voc dataset###
    elif dataset_name in voc_helen_name:
        trainset = CUB_VOC(voc_file, dataset_name, 'ori', train=True, transform=transform, is_frac=label)
        testset = CUB_VOC(voc_file, dataset_name, 'ori', train=False, transform=val_transform, is_frac=label)
    ###celeb dataset###
    elif dataset_name == 'celeb':
        trainset = Celeb(celeb_file, training = True, transform=None, train_num=162770)                                                                                                                                            
        testset = Celeb(celeb_file, training = False, transform=None, train_num=19962)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, test_loader

def net_train():
    trainset_loader, testset_loader = get_Data(IS_TRAIN, DATANAME,BATCHSIZE)
    device = torch.device("cuda")
    if LAYERS == '50':
        net = ResNet50(NUM_CLASSES, pretrained=True)
    else:
        net = ResNet18(NUM_CLASSES, pretrained=True)
    net = nn.DataParallel(net).to(device)
    model_path = os.path.join(log_path, '%s_resnet_%s_ori' % (LAYERS, DATANAME))
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
        os.makedirs(model_path)
    else:
        os.makedirs(model_path)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.module.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=125, gamma=0.7)
    test = test_celeb if DATANAME=='celeb' else test_ori
    # Train the model
    best_acc = 0.0; save_loss = [];
    for epoch in range(0, EPOCH+1):
        scheduler.step()
        net.train()
        total_loss = 0.0;
        for batch_step, input_data in tqdm(enumerate(trainset_loader,0),total=len(trainset_loader),smoothing=0.9):
            inputs, labels = input_data
            inputs, labels = inputs.to(device), labels.long().to(device)
            optimizer.zero_grad()
            output, _ = net(inputs)
            if DATANAME != 'celeb':
                loss = criterion(output, labels)
            else:
                loss = .0
                for attribution in range(NUM_CLASSES//2):
                    loss += criterion(output[:, 2*attribution:2*attribution+2], labels[:, attribution])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        ### loss save code #####
        total_loss = float(total_loss) / len(trainset_loader)
        save_loss.append(total_loss)
        np.savez(os.path.join(model_path, 'loss.npz'), loss=np.array(save_loss))
        if epoch % 50 == 0:
            train_acc = test(net, trainset_loader, NUM_CLASSES)
            print('Epoch', epoch, 'train accuracy:%.4f' % train_acc)
            torch.save(net.state_dict(), model_path+'/model_%.3d.pth' % (epoch))
        if epoch % 1 == 0:
            acc = test(net, testset_loader, NUM_CLASSES)
            print('Epoch', epoch, 'loss: %.4f' % total_loss, 'test accuracy:%.4f' % acc)
            if acc > best_acc and epoch >= 10:
                best_acc = acc
                torch.save(net.state_dict(), model_path+'/model_%.3d_%.4f.pth' % (epoch, best_acc))
    print('Finished Training')
    return net

def get_feature():
    _, testset_test = get_Data(True, DATANAME, BATCHSIZE)
    _, testset_feature = get_Data(False, DATANAME, BATCHSIZE)
    device = torch.device("cuda")
    if not os.path.exists(pretrain_model):
        raise Exception("Not such pretrain-model!")
    if LAYERS == '50':
        net = ResNet50(NUM_CLASSES)
    else:
        net = ResNet18(NUM_CLASSES)
    net = nn.DataParallel(net).to(device)
    test = test_celeb if DATANAME=='celeb' else test_ori
    acc = test(net, testset_test, NUM_CLASSES)
    print('test acc:', acc)
    #if not os.path.exists(acc_path):
    #    os.makedirs(acc_path)
    f = open(os.path.join(acc_path, 'res'+str(LAYERS)+'_'+DATANAME+'_test.txt'), 'w+')
    f.write('%s%s\n' % ('res', str(LAYERS)))
    f.write('%s\n' % DATANAME)
    f.write('acc:%f\n' % acc)
    #if not os.path.exists(save_path):
    #   os.makedirs(save_path)
    all_feature = []
    for batch_step, input_data in tqdm(enumerate(testset_feature,0),total=len(testset_feature),smoothing=0.9):
        inputs, labels = input_data
        inputs, labels = inputs.to(device), labels.long().to(device)
        net.eval()
        output, f_map = net(inputs)
        all_feature.append(f_map.cpu().numpy())
    all_feature = np.concatenate(all_feature, axis=0)
    f.write('sample num:%d' % (all_feature.shape[0]))
    f.close()
    np.savez_compressed(save_path+LAYERS+'_resnet_'+DATANAME+'_ori.npz', f_map=all_feature[...])
    print('Finished Getting Feature!')
    return net

def test_ori(net, testdata, n_cls):
    correct, total = .0, .0
    for inputs, labels in testdata:
        inputs, labels = inputs.cuda(), labels.cuda().long()
        net.eval()
        outputs, _ = net(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    return float(correct) / total

def test_celeb(net, testdata, n_cls):
    correct, total = .0, .0
    ATTRIBUTION_NUM = n_cls//2
    running_correct = np.zeros(ATTRIBUTION_NUM)
    for inputs, labels in tqdm(testdata):
        inputs, labels = inputs.cuda(), labels.cuda().long()
        net.eval()
        outputs, _ = net(inputs)
        out = outputs.data
        total += labels.size(0)
        for attribution in range(ATTRIBUTION_NUM):
            _, predicted = torch.max(out[:, 2*attribution:2*attribution+2], 1)
            correct = (predicted == labels[:, attribution]).sum().item()
            running_correct[attribution] += correct
    attr_acc = running_correct / float(total)
    return np.mean(attr_acc)

def resnet_ori_train():
    if IS_TRAIN:
        net = net_train()
    else:
        net = get_feature()


if __name__ == '__main__':
    net = resnet_ori_train()
