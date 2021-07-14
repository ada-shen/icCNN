#!/usr/bin/env python
import torch
import torch.nn as nn
import math
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from load_utils import load_state_dict_from_url
from cub_voc import CUB_VOC
import os
from tqdm import tqdm
import shutil
from utils.utils import Cluster_loss, Multiclass_loss
import numpy as np
from Similar_Mask_Generate import SMGBlock
from SpectralClustering import spectral_clustering
from newPad2d import newPad2d

IS_TRAIN = 0        # 0/1
LAYERS = '18'
DATANAME = 'voc_multi' # voc_multi
NUM_CLASSES = 6
cub_file = '/data/sw/dataset/frac_dataset'
voc_file = '/data/sw/dataset/VOCdevkit/VOC2010/voc2010_crop'
log_path = '/data/fjq/iccnn/resnet/' # for model
save_path = '/data/fjq/iccnn/basic_fmap/resnet/'  # for get_feature
acc_path = '/data/fjq/iccnn/basic_fmap/resnet/acc/'

dataset = '%s_resnet_%s_iccnn' % (LAYERS, DATANAME)
log_path = log_path + dataset + '/'
pretrain_model = log_path + 'model_2000.pth'
BATCHSIZE = 16
LR = 0.00001
EPOCH = 3000
center_num = 16
lam1 = 0.1
lam2 = 0.1
T = 2 # T = 2 ===> do sc each epoch
F_MAP_SIZE = 196
STOP_CLUSTERING = 200
if LAYERS == '18':
    CHANNEL_NUM = 256
elif LAYERS == '50':
    CHANNEL_NUM = 1024

_all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
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
        self.pad2d = newPad2d(1)#new paddig

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
        self.pad2d = newPad2d(1)#new paddig

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
    def __init__(self, block, layers, num_classes=NUM_CLASSES, zero_init_residual=False,
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
        self.smg = SMGBlock(channel_size = CHANNEL_NUM,f_map_size=F_MAP_SIZE)
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

    def forward(self, x, eval=False):
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
        if eval:
            return x
        corre_matrix = self.smg(x)
        f_map = x
        x = self.layer4(x)
        # if eval:
        #     return x
        # corre_matrix = self.smg(x,ground_true)
        # f_map = x
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x, f_map, corre_matrix

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        pretrained_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}#'fc' not in k and 'layer4.1' not in k and
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    else:
        if pretrain_model is not None:
            print("Load pretrained model",pretrain_model)
            device = torch.device("cuda")
            model = nn.DataParallel(model).to(device)
            pretrained_dict = torch.load(pretrain_model)
            pretrained_dict = {k[k.find('.')+1:]: v for k, v in pretrained_dict.items()}
            model.load_state_dict(pretrained_dict)
    return model

def ResNet18(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock, [2,2,2,2], pretrained, progress, **kwargs)

def ResNet34(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet34', BasicBlock, [3,4,6,3], pretrained, progress, **kwargs)

def ResNet50(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet50', Bottleneck, [3,4,6,3], pretrained, progress, **kwargs)

def ResNet101(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet101', Bottleneck, [3,4,23,3], pretrained, progress, **kwargs)

def ResNet152(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet152', Bottleneck, [3,8,36,3], pretrained, progress, **kwargs)

def get_Data(is_train, dataset_name, batch_size):
    val_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    voc_helen = ['bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'helen', 'voc_multi']
    ##cub dataset###
    label = None if is_train else 0
    if not is_train:
        batch_size = 1
    if dataset_name == 'cub':
        trainset = CUB_VOC(cub_file, dataset_name, 'iccnn', train=True, transform=val_transform, is_frac=label)
        testset = CUB_VOC(cub_file, dataset_name, 'iccnn', train=False, transform=val_transform, is_frac=label)
    ###cropped voc dataset###
    elif dataset_name in voc_helen:
        trainset = CUB_VOC(voc_file, dataset_name, 'iccnn', train=True, transform=val_transform, is_frac=label)
        testset = CUB_VOC(voc_file, dataset_name, 'iccnn', train=False, transform=val_transform, is_frac=label)
    ###celeb dataset###
    #elif dataset_name == 'celeb':
    #   trainset = Celeb(training = True, transform=None)
    #   testset = Celeb(training = False, transform=None)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def net_train():
    trainset_loader, testset_loader = get_Data(IS_TRAIN, DATANAME, BATCHSIZE)
    if os.path.exists(log_path):
        shutil.rmtree(log_path);os.makedirs(log_path)
    else:
        os.makedirs(log_path)
    device = torch.device("cuda")

    if LAYERS == '18':
        net = ResNet18(pretrained=False)
    elif LAYERS == '50':
        net = ResNet50(pretrained=False)

    net = nn.DataParallel(net).to(device)
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.module.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=125, gamma=0.6)

    # Train the model
    save_loss = [];save_similatiry_loss = [];save_gt=[];save_class_loss= [];save_total_loss = [];
    cs_loss = Cluster_loss()
    mc_loss = Multiclass_loss(class_num=NUM_CLASSES)
    for epoch in range(EPOCH+1):
        if epoch % T==0 and epoch < STOP_CLUSTERING:
            with torch.no_grad():
                Ground_true, loss_mask_num, loss_mask_den = offline_spectral_cluster(net, trainset_loader)
            save_gt.append(Ground_true.cpu().numpy())
        else:
            scheduler.step()
            net.train()
            all_feature = []; total_loss = 0.0;similarity_loss = 0.0;class_loss = 0.0
            for batch_step, input_data in tqdm(enumerate(trainset_loader,0),total=len(trainset_loader),smoothing=0.9):
                inputs, labels = input_data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                output, f_map, corre = net(inputs, eval=False)

                clr_loss = criterion(output, labels)
                loss1 = cs_loss.update(corre, loss_mask_num, loss_mask_den, None)
                loss2 = mc_loss.update(f_map, loss_mask_num, labels)
                loss =  clr_loss + lam1 *loss1 + lam2*loss2
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                similarity_loss += loss1.item()
                class_loss += loss2.item()
                
            ### loss save code #####
            total_loss = float(total_loss) / len(trainset_loader)
            similarity_loss = float(similarity_loss) / len(trainset_loader)
            class_loss = float(class_loss) / len(trainset_loader)
            save_total_loss.append(total_loss)
            save_similatiry_loss.append(similarity_loss)
            save_class_loss.append(class_loss)
            acc = 0
            #if epoch % 5==0:
            #    acc = test(net, testset_loader)

            print('Epoch', epoch, 'loss: %.4f' % total_loss, 'sc_loss: %.4f' % similarity_loss, 'class_loss: %.4f' % class_loss, 'test accuracy:%.4f' % acc)
            if epoch % 100 == 0:
                torch.save(net.state_dict(), log_path+'model_%.3d.pth' % (epoch))
                np.savez(log_path+'loss_%.3d.npz'% (epoch), loss=np.array(save_total_loss), similarity_loss = np.array(save_similatiry_loss), class_loss = np.array(save_class_loss),gt=np.array(save_gt))
    print('Finished Training')

def offline_spectral_cluster(net, train_data):
    net.eval()
    f_map = []
    for inputs, labels in train_data:
        inputs, labels = inputs.cuda(), labels.cuda()
        cur_fmap= net(inputs,eval=True).detach().cpu().numpy()
        f_map.append(cur_fmap)
    f_map = np.concatenate(f_map,axis=0)
    sample, channel,_,_ = f_map.shape
    f_map = f_map.reshape((sample,channel,-1))
    mean = np.mean(f_map,axis=0)
    cov = np.mean(np.matmul(f_map-mean,np.transpose(f_map-mean,(0,2,1))),axis=0)
    diag = np.diag(cov).reshape(channel,-1)
    correlation = cov/(np.sqrt(np.matmul(diag,np.transpose(diag,(1,0))))+1e-5)+1
    ground_true, loss_mask_num, loss_mask_den = spectral_clustering(correlation,n_cluster=center_num)

    return ground_true, loss_mask_num, loss_mask_den

def get_feature():
    print('pretrain_model:', pretrain_model)
    _, testset_test = get_Data(True, DATANAME, BATCHSIZE)
    _, testset_feature = get_Data(False, DATANAME, BATCHSIZE)
    device = torch.device("cuda")
    net = None
    if LAYERS == '50':
        net = ResNet50(pretrained=False)
    elif LAYERS == '18':
        net = ResNet18(pretrained=False)

    acc = test(net, testset_test)
    f = open(acc_path+dataset+'_test.txt', 'w+')
    f.write('%s\n' % dataset)
    f.write('acc:%f\n' %acc)
    all_feature = []
    for batch_step, input_data in tqdm(enumerate(testset_feature,0),total=len(testset_feature),smoothing=0.9):
        inputs, labels = input_data
        inputs, labels = inputs.to(device), labels.to(device)
        net.eval()
        f_map = net(inputs,eval=True)
        all_feature.append(f_map.detach().cpu().numpy())
    all_feature = np.concatenate(all_feature,axis=0)
    print(all_feature.shape)
    f.write('sample num:%d' % (all_feature.shape[0]))
    f.close()
    np.savez(save_path+LAYERS+'_resnet_'+DATANAME+'iccnn_.npz', f_map=all_feature[...])
    print('Finished Operation!')

def test(net, testdata):
    correct, total = .0, .0
    for inputs, labels in testdata:
        inputs, labels = inputs.cuda(), labels.cuda()
        net.eval()
        outputs, _,_ = net(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('test acc = ',float(correct) / total)
    return float(correct) / total

def resnet_multi_train():
    if IS_TRAIN ==1:
        net_train()
    elif IS_TRAIN == 0:
        get_feature()
