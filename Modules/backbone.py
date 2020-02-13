##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#Created by: Yuemeng Li
#BE department, University of Pennsylvania
#Email: ymli@seas.upenn.edu
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


import numpy as np
import torch.nn as nn
import Modules.Blockmodule as bm
import Modules.SEmodule as se
import torch.nn.functional as F
import torch

params = {'num_channels' : 1,
        'num_filters' : 64,
        'kernel_h' : 5,
        'kernel_w' : 5,
        'kernel_c' : 1,
        'stride_conv' : 1,
        'pool' : 2,
        'stride_pool' : 2,
        #Valid options : NONE, CSE, SSE, CSSE
        'se_block' : "CSSE",
        'drop_out' : 0.1}

params3D = {'num_filters' : 64,
        'kernel_h' : 5,
        'kernel_w' : 5,
        'kernel_c' : 1,
        'stride_conv' : 1,
        'pool' : 2,
        'stride_pool' : 2,
        #Valid options : NONE, CSE, SSE, CSSE
        'se_block' : "CSSE",
        'drop_out' : 0.1}


class Backbone(nn.Module):
    def __init__(self, out_channels, num_slices, se_loss=True):
        super(Backbone, self).__init__()
        
        #this is for 2D
        self.encode1 = bm.EncoderBlock(params, se_block_type='CSSE') #input size 256 256
        params['num_channels'] = 64
        self.encode2 = bm.EncoderBlock(params, se_block_type='CSSE')
        self.encode3 = bm.EncoderBlock(params, se_block_type='CSSE')
        self.encode4 = bm.EncoderBlock(params, se_block_type='CSSE')
        
        #this is for 3D encoding
        params3D['num_channels'] = int(2*num_slices + 1)
        self.encode3D1 = bm.EncoderBlock(params3D, se_block_type='CSSE') #input size 256 256
        params3D['num_channels'] = 64
        self.encode3D2 = bm.EncoderBlock(params3D, se_block_type='CSSE')
        self.encode3D3 = bm.EncoderBlock(params3D, se_block_type='CSSE')
        self.encode3D4 = bm.EncoderBlock(params3D, se_block_type='CSSE')
        self.bottleneck3D = bm.DenseBlock(params3D, se_block_type='CSSE') #output size 16 16
        
        #this is after concat
        self.bottleneck = bm.DenseBlock(params, se_block_type='CSSE') #output size 16 16
        params['num_channels'] = 192 #unpool has concat, 64+64
        self.decode4 = bm.DecoderBlock(params, se_block_type='CSSE')
        params['num_channels'] = 128
        self.decode3 = bm.DecoderBlock(params, se_block_type='CSSE') #output size 64 64
        self.decode2 = bm.DecoderBlock(params, se_block_type='CSSE')
        self.decode1 = bm.DecoderBlock(params, se_block_type='CSSE') #label decoder
        self.decode0 = bm.DecoderBlock(params, se_block_type='CSSE') #skull decoder
        
        self.encmodule = EncModule(64, out_channels, ncodes=32,
                                   se_loss=se_loss)
        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False),
                                   nn.Conv2d(64, out_channels, 1)) #label output
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False),
                                   nn.Conv2d(64, 2, 1)) #skull output
        self.conv8 = nn.Conv2d(64, out_channels, 1) #merged label output
        
        

    def forward(self, input, input3D):
        """
        :param input: X
        :return: probabiliy map
        """
        #for 2D
        e1, out1, ind1 = self.encode1.forward(input)
        e2, out2, ind2 = self.encode2.forward(e1)
        e3, out3, ind3 = self.encode3.forward(e2)
        e4, out4, ind4 = self.encode4.forward(e3)
        
        #for 3D
        e13D, _, _ = self.encode3D1.forward(input3D)
        e23D, _, _ = self.encode3D2.forward(e13D)
        e33D, _, _ = self.encode3D3.forward(e23D)
        e43D, _, ind43D = self.encode3D4.forward(e33D)

        bn = self.bottleneck.forward(e4)
        bn3D = self.bottleneck3D(e43D)
        
        bn_dense = torch.cat((bn, bn3D), dim =1)
        ind_desne = torch.cat((ind4, ind43D), dim =1)
        
        d4 = self.decode4.forward(bn_dense, out4, ind_desne)
        d3 = self.decode3.forward(d4, out3, ind3)
        d2 = self.decode2.forward(d3, out2, ind2)
        
        #for label output
        d1 = self.decode1.forward(d2, out1, ind1)
        
        #for skull output
        d0 = self.decode0.forward(d2, out1, ind1)
        
        upfeat, seout = self.encmodule(bn_dense, d1)
        
        out_label = self.conv6(upfeat)
        out_skull = self.conv7(d0)
        
        pre_fuse = upfeat * d0
        out_fuse = self.conv8(pre_fuse)
        
        return tuple([out_fuse, out_label, out_skull, seout])


class EncModule(nn.Module):
    def __init__(self, in_channels, nclass, ncodes=32, se_loss=True):
        super(EncModule, self).__init__()
        self.se_loss = se_loss
        self.encoding = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid())
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if self.se_loss:
            self.selayer = nn.Linear(in_channels, nclass)

    def forward(self, x, upfeat):
        en = self.encoding(x)
        b, c, _, _ = en.size()
        gamma = self.avg_pool(en).view(b, c)
        y = self.fc(gamma).view(b, c, 1, 1)
        outputs = [F.relu_(upfeat + upfeat * y)]
        if self.se_loss:
            outputs.append(self.selayer(gamma))
        return tuple(outputs)
