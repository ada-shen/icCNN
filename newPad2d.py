import torch
from torch.nn import Module
import torch.nn as nn
import copy

class newPad2d(Module):
    def __init__(self,length):
        super(newPad2d,self).__init__()
        self.length = length
        self.zeroPad = nn.ZeroPad2d(self.length)

    def forward(self, input):
        b,c,h,w = input.shape
        output = self.zeroPad(input)

        #output = torch.FloatTensor(b,c,h+self.length*2,w+self.length*2)
        #output[:,:,self.length:self.length+h,self.length:self.length+w] = input

        for i in range(self.length):
        # 一层的四个切片
            output[:, :, self.length:self.length+h, i] = output[:, :, self.length:self.length+h, self.length]
            output[:, :, self.length:self.length + h, w+ self.length+i] = output[:, :, self.length:self.length + h,
                                                                self.length-1+w]
            output[:, :, i, self.length:self.length+w] = output[:, :, self.length, self.length:self.length+w]
            output[:, :, h+self.length+i, self.length:self.length + w] = output[:, :, h + self.length-1,
                                                                self.length:self.length + w]
         # 对角进行特别处理
        for j in range(self.length):
            for k in range(self.length):
                output[:,:,j,k]=output[:,:,self.length,self.length]
                output[:, :, j, w+ self.length+k] = output[:, :, self.length, self.length-1+w]
                output[:, :, h+self.length+j, k] = output[:, :, h + self.length-1, self.length]
                output[:, :, h+self.length+j, w + self.length + k] = output[:, :, h + self.length-1, self.length - 1 + w]
        return output
'''
class newPad2d(Module):
    def __init__(self,length):
        super(newPad2d,self).__init__()
        self.length = length
        self.zeroPad = nn.ZeroPad2d(self.length)

    def forward(self, input):
        b,c,h,w = input.shape
        output = self.zeroPad(input)
        out_cp = torch.zeros_like(output)
        #output = torch.FloatTensor(b,c,h+self.length*2,w+self.length*2)
        #output[:,:,self.length:self.length+h,self.length:self.length+w] = input

        # 一层的四个切片
        out_cp[:, :, self.length:self.length+h, 0:self.length] = output[:, :, self.length:self.length+h, self.length].view(b,c,h,1).repeat(1,1,1,self.length)
        out_cp[:, :, self.length:self.length + h, w+self.length: 2*self.length+w] = output[:, :, self.length:self.length + h, self.length-1+w].view(b,c,h,1).repeat(1,1,1,self.length)
        out_cp[:, :, 0:self.length, self.length:self.length+w] = output[:, :, self.length, self.length:self.length+w].view(b,c,1,w).repeat(1,1,self.length,1)
        out_cp[:, :, h+self.length:h+2*self.length, self.length:self.length+w] = output[:, :, h + self.length-1, self.length:self.length + w].view(b,c,1,w).repeat(1,1,self.length,1)
        # 对角进行特别处理
        out_cp[:,:, 0:self.length, 0:self.length] = output[:,:,self.length,self.length].view(b,c,1,1).repeat(1,1,self.length,self.length)
        out_cp[:, :, 0:self.length, w+self.length: 2*self.length+w] = output[:, :, self.length, self.length-1+w].view(b,c,1,1).repeat(1,1,self.length,self.length)
        out_cp[:, :, h+self.length:h+2*self.length, 0:self.length] = output[:, :, h + self.length-1, self.length].view(b,c,1,1).repeat(1,1,self.length,self.length)
        out_cp[:, :, h+self.length:h+2*self.length, w+self.length: 2*self.length+w] = output[:, :, h + self.length-1, self.length - 1 + w].view(b,c,1,1).repeat(1,1,self.length,self.length)
        out_cp[:, :, self.length:self.length+h, self.length:w+self.length] = output[:, :, self.length:self.length+h, self.length:w+self.length]
        return out_cp
'''
