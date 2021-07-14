#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import math
import torchvision.transforms as transforms
import torchvision as tv
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
LAYERS = '13'
DATANAME = 'voc_multi' # voc_multi
NUM_CLASSES = 6
cub_file = '/data/sw/dataset/frac_dataset'
voc_file = '/data/sw/dataset/VOCdevkit/VOC2010/voc2010_crop'
log_path = '/data/fjq/iccnn/vgg/' # for model
save_path = '/data/fjq/iccnn/basic_fmap/vgg/'  # for get_feature
acc_path = '/data/fjq/iccnn/basic_fmap/vgg/acc/'

dataset = '%s_vgg_%s_iccnn' % (LAYERS, DATANAME)
log_path = log_path + dataset + '/'
pretrain_model = log_path + 'model_2000.pth'
BATCHSIZE = 1
LR = 0.000001
EPOCH = 3000
center_num = 16
lam1 = 0.1
lam2 = 0.1
T = 2 # T = 2 ===> do sc each epoch
F_MAP_SIZE = 196
STOP_CLUSTERING = 200
if LAYERS == '13':
    CHANNEL_NUM = 512
elif LAYERS == '16':
    CHANNEL_NUM = 512

__all__ = ['VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn','vgg19_bn', 'vgg19',]
cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],}
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',}

class VGG(nn.Module):
    def __init__(self, features, num_classes=NUM_CLASSES, init_weights=True, cfg=None): 
        super(VGG, self).__init__()
        self.features = features
        # define the layer number of the relu after the top conv layer of VGG
        if cfg=='D': # VGG16
            self.target_layer = 42
        if cfg=='B': # VGG13
            self.target_layer = 33
        self.layer_num = self.target_layer
        self.smg = SMGBlock(channel_size = CHANNEL_NUM, f_map_size=F_MAP_SIZE)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.pad2d = newPad2d(1)
        self.classifier = nn.Sequential( 
            #fc6
            nn.Linear(512*7*7, 4096),nn.ReLU(True),nn.Dropout(0.5),
            #fc7
            nn.Linear(4096, 512),nn.ReLU(True),nn.Dropout(0.5),
            #fc8
            nn.Linear(512, num_classes))
       
        if init_weights:
            self._initialize_weights()
        
    def forward(self, x, eval=False):
        for layer in self.features[:self.target_layer+1]:
            if isinstance(layer,nn.Conv2d):
                x = self.pad2d(x)
            x = layer(x)
        if eval:
            return x
        corre_matrix = self.smg(x)
        f_map = x.detach()
        for layer in self.features[self.target_layer+1:]:
            if isinstance(layer,nn.Conv2d):
                x = self.pad2d(x)
            x = layer(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x, f_map, corre_matrix

    def _initialize_weights(self):
        for layer, m in enumerate(self.modules()):
            #print(layer,m)
            if layer > self.layer_num:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, 3, padding=0)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def vgg16(arch, cfg, device=None, pretrained=False, progress=True, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
        kwargs['cfg'] = cfg
    model = VGG(make_layers(cfgs[cfg], batch_norm=True), **kwargs)
    if pretrained:
        if pretrain_model is None:
            state_dict = load_state_dict_from_url(model_urls[arch],progress=progress)
            pretrained_dict = {k: v for k, v in state_dict.items() if 'classifier' not in k}
            model_dict = model.state_dict()
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        else:
            device = torch.device("cuda")
            model = nn.DataParallel(model).to(device)
            pretrained_dict = torch.load(pretrain_model)
            if IS_TRAIN == 0:
                pretrained_dict = {k[k.find('.')+1:]: v for k, v in pretrained_dict.items()}
            # model_dict = model.state_dict()
            # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # model_dict.update(pretrained_dict)
            model.load_state_dict(pretrained_dict)

    if device is not None:
        model = nn.DataParallel(model).to(device)
    return model

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
    #    trainset = Celeb(training = True, transform=None)
    #    testset = Celeb(training = False, transform=None)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def net_train():
    trainset_loader, testset_loader = get_Data(IS_TRAIN, DATANAME,  BATCHSIZE)

    if os.path.exists(log_path):
        shutil.rmtree(log_path);os.makedirs(log_path)
    else:
        os.makedirs(log_path)
    device = torch.device("cuda")

    net = None
    if LAYERS == '13':
        net = vgg16(arch='vgg13_bn',cfg='B', device=device, pretrained=True, progress=True)
    elif LAYERS == '16':
        net = vgg16(arch='vgg16_bn',cfg='D', device=device, pretrained=True, progress=True)  

    # Loss and Optimizer
    criterion_ce = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.module.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=125, gamma=0.6)

    # Train the model
    best_acc = 0.0; save_total_loss = []; save_similatiry_loss = [];save_gt=[]
    save_class_loss= []
    save_num = []
    save_label = []
    cs_loss = Cluster_loss()
    mc_loss = Multiclass_loss(class_num= NUM_CLASSES)
    for epoch in range(EPOCH+1):
        if epoch % T==0 and epoch < STOP_CLUSTERING:
            with torch.no_grad():
                Ground_true, loss_mask_num, loss_mask_den = offline_spectral_cluster(net, trainset_loader)
            save_gt.append(Ground_true.cpu().numpy())
        else:
            scheduler.step()
            net.train()
            all_feature = []
            total_loss = 0.0
            similarity_loss = 0.0
            class_loss = 0.0
            for batch_step, input_data in tqdm(enumerate(trainset_loader,0),total=len(trainset_loader),smoothing=0.9):
                inputs, labels = input_data
                inputs, labels = inputs.to(device), labels.to(device)

                output, f_map, corre = net(inputs)
                clr_loss = criterion_ce(output, labels)
                loss1 = cs_loss.update(corre, loss_mask_num, loss_mask_den, None)
                loss2 = mc_loss.update(f_map, loss_mask_num, labels)
                loss =  clr_loss + lam1 *loss1 + lam2*loss2
                optimizer.zero_grad()
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
            #acc = test(net, testset_loader)
            acc = 0
            print('Epoch', epoch, 'loss: %.4f' % total_loss, 'sc_loss: %.4f' % similarity_loss, 'class_loss: %.4f' % class_loss, 'test accuracy:%.4f' % acc)
            if epoch % 100 == 0:
                torch.save(net.state_dict(), log_path+'model_%.3d.pth' % (epoch))
                np.savez(log_path+'loss_%.3d.npz'% (epoch), loss=np.array(save_total_loss), similarity_loss = np.array(save_similatiry_loss), class_loss = np.array(save_class_loss),gt=np.array(save_gt))

    print('Finished Training')
    return net

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

def test(net, testdata):
    correct, total = .0, .0
    for inputs, labels in testdata:
        inputs, labels = inputs.cuda(), labels.cuda()
        net.eval()
        outputs, _,_ = net(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    return float(correct) / total

def get_feature():
    print('pretrain_model:', pretrain_model)
    _,testset_test = get_Data(True, DATANAME, BATCHSIZE)
    device = torch.device("cuda")
    net = None
    if LAYERS == '13':
        net = vgg16(arch='vgg13_bn',cfg='B', device=device, pretrained=True, progress=True)
    elif LAYERS == '16':
        net = vgg16(arch='vgg16_bn',cfg='D', device=device, pretrained=True, progress=True)

    net = nn.DataParallel(net).to(device)
    acc = test(net, testset_test)##
    f = open(acc_path+dataset+'_test.txt', 'w+')
    f.write('%s\n' % dataset)
    f.write('acc:%f\n' % acc)
    print(acc)
    all_feature = []
    for batch_step, input_data in tqdm(enumerate(testset_test,0),total=len(testset_test),smoothing=0.9):
        inputs, labels = input_data
        inputs, labels = inputs.to(device), labels.to(device)
        net.eval()
        f_map = net(inputs,eval=True)
        all_feature.append(f_map.detach().cpu().numpy())
    all_feature = np.concatenate(all_feature,axis=0)
    f.write('sample num:%d' % (all_feature.shape[0]))
    f.close()
    print(all_feature.shape)
    np.savez(save_path+LAYERS+'_vgg_'+DATANAME+'_iccnn.npz', f_map=all_feature[...])
    print('Finished Operation!')
    return net

def vgg_multi_train():
    if IS_TRAIN == 1:
        net_train()
    elif IS_TRAIN == 0:
        get_feature()
