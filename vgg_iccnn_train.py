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
import numpy as np
from celeb import Celeb
from Similar_Mask_Generate import SMGBlock
from SpectralClustering import spectral_clustering
from utils.utils import Cluster_loss
from newPad2d import newPad2d

IS_TRAIN = 0        # 0/1
LAYERS = '13'
DATANAME = 'bird'
NUM_CLASSES = 80 if DATANAME == 'celeb' else 2
cub_file = '/data/sw/dataset/frac_dataset'
voc_file = '/data/sw/dataset/VOCdevkit/VOC2010/voc2010_crop'
celeb_file = '/home/user05/fjq/dataset/CelebA/'
log_path = '/data/fjq/iccnn/vgg/' # for model
save_path = '/data/fjq/iccnn/basic_fmap/vgg/'  # for get_feature
acc_path = '/data/fjq/iccnn/basic_fmap/vgg/acc/'

dataset = '%s_vgg_%s_iccnn' % (LAYERS, DATANAME)
log_path = log_path + dataset + '/'
pretrain_model = log_path + 'model_2000.pth'
BATCHSIZE = 1
LR = 0.00001
EPOCH = 2500
center_num = 5
lam = 0.1
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
    def __init__(self, features, num_classes, init_weights=True, cfg=None): 
        super(VGG, self).__init__()
        
        self.features = features
        if cfg=='D': # VGG16
            self.target_layer = 42
        if cfg=='B': # VGG13
            self.target_layer = 33
        self.layer_num = self.target_layer
        self.pad2d = newPad2d(1) #nn.ReplicationPad2d(1)
        self.smg = SMGBlock(channel_size = CHANNEL_NUM, f_map_size=F_MAP_SIZE)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
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

def vgg16(arch, cfg, num_class, device=None, pretrained=False, progress=True, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
        kwargs['cfg'] = cfg
    model = VGG(make_layers(cfgs[cfg], batch_norm=True), num_class, **kwargs)
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
    celeb_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
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
    elif dataset_name == 'celeb':
        trainset = Celeb(celeb_file, training = True, transform=celeb_transform, train_num=10240)
        testset = Celeb(celeb_file, training = False, transform=celeb_transform, train_num=19962)
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
    test = test_celeb if DATANAME=='celeb' else test_ori

    net = None
    if LAYERS == '13':
        net = vgg16(arch='vgg13_bn',cfg='B', num_class=NUM_CLASSES, device=device, pretrained=True, progress=True)
    elif LAYERS == '16':
        net = vgg16(arch='vgg16_bn',cfg='D', num_class=NUM_CLASSES, device=device, pretrained=True, progress=True)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.module.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=125, gamma=0.6)

    # Train the model
    best_acc = 0.0
    save_total_loss = []; save_similatiry_loss = [];save_gt=[]
    cs_loss = Cluster_loss()
    for epoch in range(EPOCH+1):
        if (epoch) % T==0 and epoch < STOP_CLUSTERING:
            with torch.no_grad():
                Ground_true, loss_mask_num, loss_mask_den = offline_spectral_cluster(net, trainset_loader, DATANAME)
            save_gt.append(Ground_true.cpu().numpy())
        else:
            scheduler.step()
            net.train()
            total_loss = 0.0;similarity_loss = 0.0

            for batch_step, input_data in tqdm(enumerate(trainset_loader,0),total=len(trainset_loader),smoothing=0.9):
                inputs, labels = input_data
                inputs, labels = inputs.to(device), labels.long().to(device)
                optimizer.zero_grad()
                output, f_map, corre = net(inputs, eval=False)

                if DATANAME != 'celeb':
                    clr_loss = criterion(output, labels)
                else:
                    clr_loss = .0
                    for attribution in range(NUM_CLASSES//2):
                        clr_loss += criterion(output[:, 2*attribution:2*attribution+2], labels[:, attribution])    
                    labels = None                

                loss_ = cs_loss.update(corre, loss_mask_num, loss_mask_den, labels)
                loss =  clr_loss + lam * loss_
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                similarity_loss += loss_.item()
                
            ### loss save code #####
            total_loss = float(total_loss) / len(trainset_loader)
            similarity_loss = float(similarity_loss) / len(trainset_loader)
            save_total_loss.append(total_loss)
            save_similatiry_loss.append(similarity_loss)
            acc = 0#test(net, testset_loader, n_cls)
            print('Epoch', epoch, 'loss: %.4f' % total_loss,'sc_loss: %.4f' % similarity_loss, 'test accuracy:%.4f' % acc)
        
        if epoch % 100 == 0 :
            torch.save(net.state_dict(), log_path+'model_%.3d.pth' % (epoch))
            np.savez(log_path+'loss_%.3d.npz'% (epoch), loss=np.array(save_total_loss), similarity_loss = np.array(save_similatiry_loss),gt=np.array(save_gt))
        if epoch %1 == 0:
            if acc > best_acc:
                best_acc = acc
                torch.save(net.state_dict(), log_path+'model_%.3d_%.4f.pth' % (epoch,best_acc))
    print('Finished Training')
    return net

def offline_spectral_cluster(net, train_data, dataname):
    net.eval()
    f_map = []
    for inputs, labels in train_data:
        inputs, labels = inputs.cuda(), labels.cuda()
        cur_fmap= net(inputs,eval=True).detach().cpu().numpy()
        f_map.append(cur_fmap)
        if dataname == 'celeb' and len(f_map)>=1024:
            break
    f_map = np.concatenate(f_map,axis=0)
    sample, channel,_,_ = f_map.shape
    f_map = f_map.reshape((sample,channel,-1))
    mean = np.mean(f_map,axis=0)
    cov = np.mean(np.matmul(f_map-mean,np.transpose(f_map-mean,(0,2,1))),axis=0)
    diag = np.diag(cov).reshape(channel,-1)
    correlation = cov/(np.sqrt(np.matmul(diag,np.transpose(diag,(1,0))))+1e-5)+1
    ground_true, loss_mask_num, loss_mask_den = spectral_clustering(correlation,n_cluster=center_num)
    return ground_true, loss_mask_num, loss_mask_den

def test_ori(net, testdata, n_cls):
    correct, total = .0, .0
    for inputs, labels in tqdm(testdata):
        inputs, labels = inputs.cuda(), labels.cuda().long()
        net.eval()
        outputs, _, _ = net(inputs)
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
        outputs, _, _ = net(inputs)
        out = outputs.data
        total += labels.size(0)
        for attribution in range(ATTRIBUTION_NUM):
            _, predicted = torch.max(out[:, 2*attribution:2*attribution+2], 1)
            correct = (predicted == labels[:, attribution]).sum().item()
            running_correct[attribution] += correct
    attr_acc = running_correct / float(total)
    return np.mean(attr_acc)


def get_feature():
    print('pretrain_model:', pretrain_model)
    _,testset_test = get_Data(True, DATANAME, BATCHSIZE)
    _,testset_feature = get_Data(False, DATANAME, BATCHSIZE)
    device = torch.device("cuda")
    net = None
    if LAYERS == '13':
        net = vgg16(arch='vgg13_bn',cfg='B', num_class=NUM_CLASSES, device=device, pretrained=True, progress=True)
    elif LAYERS == '16':
        net = vgg16(arch='vgg16_bn',cfg='D', num_class=NUM_CLASSES, device=device, pretrained=True, progress=True)

    net = nn.DataParallel(net).to(device)
    test = test_celeb if DATANAME=='celeb' else test_ori
    acc = test(net, testset_test, NUM_CLASSES)
    f = open(acc_path+dataset+'_test.txt', 'w+')
    f.write('%s\n' % dataset)
    f.write('acc:%f\n' % acc)
    print(acc)
    all_feature = []
    for batch_step, input_data in tqdm(enumerate(testset_feature,0),total=len(testset_feature),smoothing=0.9):
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

def vgg_single_train():
    if IS_TRAIN == 1:
        net_train()
    elif IS_TRAIN == 0:
        get_feature()

