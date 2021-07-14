#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch

class EMA_FM(): 
    def __init__(self, decay=0.9, first_decay=0.0, channel_size=512, f_map_size=196, is_use = False):
        self.decay = decay
        self.first_decay = first_decay
        self.is_use = is_use
        self.shadow = {}
        self.epsional = 1e-5
        if is_use:
            self._register(channel_size=channel_size, f_map_size= f_map_size)

    def _register(self, channel_size=512, f_map_size=196):
        Init_FM = torch.zeros((f_map_size, channel_size),dtype=torch.float)
        self.shadow['FM'] = Init_FM.cuda().clone()
        self.is_first = True

    def update(self, input):
        B, C, _ = input.size()
        if not(self.is_use):
            return torch.ones((C,C), dtype=torch.float)
        decay = self.first_decay if self.is_first else self.decay
        ####### FEATURE SIMILARITY MATRIX EMA ########
        # Mu = torch.mean(input,dim=0)   
        self.shadow['FM'] = (1.0 - decay) * torch.mean(input,dim=0) + decay * self.shadow['FM']
        self.is_first = False
        return self.shadow['FM']

class Cluster_loss():
    def __init__(self):
        pass

    def update(self, correlation, loss_mask_num, loss_mask_den, labels):
        batch, channel, _ = correlation.shape
        c, _, _ = loss_mask_num.shape
        if labels is not None:
            label_mask = (1 - labels).view(batch, 1, 1)
            ## smg_loss if only available for positive sample
            correlation = correlation * label_mask
        correlation = (correlation / batch).view(1, batch, channel, channel).repeat(c, 1, 1, 1)

        new_Num = torch.sum(correlation * loss_mask_num.view(c, 1, channel, channel).repeat(1, batch, 1, 1),
                            dim=(1, 2, 3))
        new_Den = torch.sum(correlation * (loss_mask_den).view(c, 1, channel, channel).repeat(1, batch, 1, 1),
                            dim=(1, 2, 3))
        ret_loss = -torch.sum(new_Num / (new_Den + 1e-5))
        return ret_loss

class Multiclass_loss():
    def __init__(self, class_num=None):
        self.class_num = class_num

    def get_label_mask(self, label):
        label = label.cpu().numpy()
        sz = label.shape[0]
        label_mask_num = []
        label_mask_den = []
        for i in range(self.class_num):
            idx = np.where(label == i)[0]
            cur_mask_num = np.zeros((sz, sz))
            cur_mask_den = np.zeros((sz, sz))
            for j in idx:
                cur_mask_num[j][idx] = 1
                cur_mask_den[j][:] = 1
            label_mask_num.append(np.expand_dims(cur_mask_num, 0))
            label_mask_den.append(np.expand_dims(cur_mask_den, 0))
        label_mask_num = np.concatenate(label_mask_num, axis=0)
        label_mask_den = np.concatenate(label_mask_den, axis=0)
        return torch.from_numpy(label_mask_num).float().cuda(), torch.from_numpy(label_mask_den).float().cuda()

    def update(self, fmap, loss_mask_num, label):
        B, C, _, _ = fmap.shape
        center, _, _ = loss_mask_num.shape
        fmap = fmap.view(1, B, C, -1).repeat(center, 1, 1, 1)
        mean_activate = torch.mean(torch.matmul(loss_mask_num.view(center, 1, C, C).repeat(1, B, 1, 1), fmap),
                                   dim=(2, 3))
        # cosine
        mean_activate = torch.div(mean_activate, torch.norm(mean_activate, p=2, dim=0, keepdim=True) + 1e-5)
        inner_dot = torch.matmul(mean_activate.permute(1, 0), mean_activate).view(-1, B, B).repeat(self.class_num, 1, 1)
        label_mask, label_mask_intra = self.get_label_mask(label)

        new_Num = torch.mean(inner_dot * label_mask, dim=(1, 2))
        new_Den = torch.mean(inner_dot * label_mask_intra, dim=(1, 2))
        ret_loss = -torch.sum(new_Num / (new_Den + 1e-5))
        return ret_loss

def Cal_Center(fmap, gt):
    f_1map = fmap.detach().cpu().numpy()
    matrix = gt.detach().cpu().numpy()
    B, C, H, W = f_1map.shape
    cluster = []
    visited = np.zeros(C)
    for i in range(matrix.shape[0]):
        tmp = []
        if(visited[i]==0):
            for j in range(matrix.shape[1]):
                if(matrix[i][j]==1 ):
                    tmp.append(j)
                    visited[j]=1;
            cluster.append(tmp)
    center = []
    for i in range(len(cluster)):
        cur_clustet_fmap = f_1map[:,cluster[i],...]
        cluster_center = np.mean(cur_clustet_fmap,axis=1)
        center.append(cluster_center)
    center = np.transpose(np.array(center),[1,0,2,3])
    center = torch.from_numpy(center).float()
    return center
