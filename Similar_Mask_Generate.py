"""
Created on 03 05 2020
@author: H
"""
import torch
from torch import nn
from torch.nn import functional as F
from SpectralClustering import spectral_clustering
from utils.utils import EMA_FM

class SMGBlock(nn.Module):
    def __init__(self, channel_size=2048, f_map_size=196):
        super(SMGBlock, self).__init__()

        self.EMA_FM = EMA_FM(decay=0.95, first_decay=0.0, channel_size=channel_size, f_map_size=f_map_size, is_use=True)


    def forward(self, x):
        '''
        :param x: (b, c, h, w)
        :return:
        '''
        batch_size, channel, _,_ = x.size()
        theta_x = x.view(batch_size,channel,-1).permute(0,2,1).contiguous()
        transpose_x = x.view(batch_size,channel,-1).permute(0,2,1).contiguous()# [b,h×w,c]
        with torch.no_grad():
            f_mean = self.EMA_FM.update(theta_x)
        sz = f_mean.size()[0]
        f_mean = f_mean.view(1,channel,sz)
        f_mean_transposed = f_mean.permute(0,2,1)
        Local = torch.matmul(theta_x.permute(0, 2, 1)-f_mean, theta_x-f_mean_transposed)
        diag = torch.eye(channel).view(-1,channel,channel).cuda()
        cov = torch.sum(Local*diag,dim=2).view(batch_size,channel,1)
        cov_transpose = cov.permute(0,2,1)
        norm = torch.sqrt(torch.matmul(cov,cov_transpose))
        correlation = torch.div(Local,norm)+1 ## normlize to [0,2]

        return correlation

def bn(input,eps=1e-5):
    # input b,c,n
    inSize = input.size()
    mean = input.mean(dim=0)##.view(inSize[0],-1)
    std = input.std(dim=0)#.view(inSize[0],-1)
    y = torch.div(input-mean,std+eps)
    return y

def fn(input,eps=1e-5):
    # input b,c,n
    inSize = input.size()
    mean = input.view(inSize[0],-1).mean(dim=-1)
    std = input.view(inSize[0],-1).std(dim=-1)
    y = torch.div(input-mean.view(inSize[0],1,1),std.view(inSize[0],1,1)+eps)
    return y

def single_max_min_norm(input,eps=1e-5):
    # input b,c,n
    inSize = input.size()
    max_ = torch.max(input.view(inSize[0],-1),-1)[0]
    min_ = torch.min(input.view(inSize[0],-1),-1)[0]
    #print(min_.shape)
    y = torch.div(input-min_.view(inSize[0],1,1),max_.view(inSize[0],1,1)-min_.view(inSize[0],1,1)+eps)
    return y

def batch_max_min_norm(input,eps=1e-5):
    # input b,c,n
    inSize = input.size()
    input_p = input.permute(1,0,2).contiguous()
    max_ = torch.max(input_p.view(inSize[1],-1),-1)[0]
    min_ = torch.min(input_p.view(inSize[1],-1),-1)[0]
    #print(min_.shape)
    y = torch.div(input-min_.view(1,inSize[1],1),max_.view(1,inSize[1],1)-min_.view(1,inSize[1],1)+eps)
    return y

if __name__ == '__main__':
    import torch

    img = torch.zeros(1, 1024, 14, 14)
    net = SMGBlock(1024, 196)
    out = net(img)
    print(out.size())



