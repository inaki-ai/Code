import torch
from torch import nn
from dropblock import DropBlock2D, LinearScheduler

try:
    from common.utils import *
    from models.segmentors.pooling_layers import HartleyPool2d, HybridPooling
except:
    pass

class DSC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DSC, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=(kernel_size-1)//2, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
    

class InteptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InteptionBlock, self).__init__()
        
        self.conv_k5 = nn.Sequential(
            DSC(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            DSC(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            DSC(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
    


class InceptionBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, pooling='max'):
        super().__init__()
        
        self.activation = nn.SiLU(inplace=True)

        self.convd_k1 = nn.Sequential(
            DSC(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        
        self.convd_k3 = nn.Sequential(
            DSC(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        
        self.convd_k5 = nn.Sequential(
            DSC(in_channels, out_channels, kernel_size=5),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        
        if pooling == 'max':
            self.pooling = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                DSC(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True)
            )
        elif pooling == 'max':
            self.pooling = nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                DSC(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True)
            )
        
    def forward(self, x):

        k1 = self.convd_k1(x)
        k3 = self.convd_k3(x)
        k5 = self.convd_k5(x)
        k7 = self.pooling(x)
        
        out = k1 + k3 + k5 + k7

        return out
    
class ResidualInceptionBlock(nn.Module):
    
    def __init__(self, in_channels, mid_channels, out_channels, dropout='none', dropout_p=0.04, block_size=3):
        super().__init__()
        
        self.dropout = dropout
        
        if dropout == 'dropout':
            self.dropout_layer = nn.Dropout(p=dropout_p)
        elif dropout == 'dropblock':
            self.dropout_layer = LinearScheduler(
                DropBlock2D(block_size=block_size, drop_prob=0.),
                start_value=0.,
                stop_value=dropout_p,
                nr_steps=10
            )
        
        self.subblock1 = InceptionBlock(in_channels, mid_channels)
        self.subblock2 = InceptionBlock(mid_channels, out_channels)
        
        self.convd_k1 = nn.Sequential(
            DSC(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )
        
    def forward(self, x):
        
        out = self.subblock1(x)
        out = self.subblock2(out)
        
        if self.dropout == 'dropout' or self.dropout == 'dropblock':
            return self.dropout_layer(out + self.convd_k1(x))
        else:
            return out + self.convd_k1(x)
        
        
class UnetSharp(nn.Module):

    def __init__(self, input_channels=1, nc=16, img_size=128, pooling="Max", dropout='dropout', block_size=3):
        super().__init__()

        dropout_ps = [0.075, 0.027, 0.006, 0.002, 0.075]
        dropout_ps = [0.01, 0.075, 0.027, 0.006, 0.002, 0.075]
        dropblock_sizes = [17, 11, 5, 3]
        
        self.pooling = pooling
        
        if self.pooling == 'Max':
            self.pool = nn.MaxPool2d(2, 2)
        elif self.pooling == 'Hartley':
            self.hartley_pool_x1_0 = HartleyPool2d(img_size//2)
            self.hartley_pool_x2_0 = HartleyPool2d(img_size//4)
            self.hartley_pool_x3_0 = HartleyPool2d(img_size//8)
            self.hartley_pool_x4_0 = HartleyPool2d(img_size//16)
        elif self.pooling == 'Hybrid':
            self.hybrid_pool_x1_0 = HybridPooling(img_size//2, nc)
            self.hybrid_pool_x2_0 = HybridPooling(img_size//4, nc*2)
            self.hybrid_pool_x3_0 = HybridPooling(img_size//8, nc*4)
            self.hybrid_pool_x4_0 = HybridPooling(img_size//16, nc*8)
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = ResidualInceptionBlock(input_channels, nc, nc, dropout=dropout, dropout_p=dropout_ps[0], block_size=dropblock_sizes[0])
        self.conv1_0 = ResidualInceptionBlock(nc, nc*2, nc*2, dropout=dropout, dropout_p=dropout_ps[1], block_size=dropblock_sizes[1])
        self.conv2_0 = ResidualInceptionBlock(nc*2, nc*4, nc*4, dropout=dropout, dropout_p=dropout_ps[2], block_size=dropblock_sizes[2])
        self.conv3_0 = ResidualInceptionBlock(nc*4, nc*8, nc*8, dropout=dropout, dropout_p=dropout_ps[3], block_size=dropblock_sizes[3])
        self.conv4_0 = ResidualInceptionBlock(nc*8, nc*16, nc*16, dropout=dropout, dropout_p=dropout_ps[4])

        self.conv0_1 = ResidualInceptionBlock(nc+nc*2, nc, nc, dropout=dropout, dropout_p=dropout_ps[0], block_size=dropblock_sizes[0])
        self.conv1_1 = ResidualInceptionBlock(nc*2+nc*4, nc*2, nc*2, dropout=dropout, dropout_p=dropout_ps[1], block_size=dropblock_sizes[1])
        self.conv2_1 = ResidualInceptionBlock(nc*4+nc*8, nc*4, nc*4, dropout=dropout, dropout_p=dropout_ps[2], block_size=dropblock_sizes[2])
        self.conv3_1 = ResidualInceptionBlock(nc*8+nc*16, nc*8, nc*8, dropout=dropout, dropout_p=dropout_ps[3], block_size=dropblock_sizes[3])

        self.conv0_2 = ResidualInceptionBlock(nc*2+nc*2, nc, nc, dropout=dropout, dropout_p=dropout_ps[0], block_size=dropblock_sizes[0])
        self.conv1_2 = ResidualInceptionBlock(nc*2*2+nc*4, nc*2, nc*2, dropout=dropout, dropout_p=dropout_ps[1], block_size=dropblock_sizes[1])
        self.conv2_2 = ResidualInceptionBlock(nc*4*2+nc*8, nc*4, nc*4, dropout=dropout, dropout_p=dropout_ps[2], block_size=dropblock_sizes[2])

        self.conv0_3 = ResidualInceptionBlock(nc*3+nc*2, nc, nc, dropout=dropout, dropout_p=dropout_ps[0], block_size=dropblock_sizes[0])
        self.conv1_3 = ResidualInceptionBlock(nc*2*3+nc*4, nc*2, nc*2, dropout=dropout, dropout_p=dropout_ps[1], block_size=dropblock_sizes[1])

        self.conv0_4 = ResidualInceptionBlock(nc*4+nc*2, nc, nc, dropout=dropout, dropout_p=dropout_ps[0], block_size=dropblock_sizes[0])

        self.final = DSC(nc, 1, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        if self.pooling == 'Max':
            x1_0 = self.conv1_0(self.pool(x0_0))
        elif self.pooling == 'Hartley':
            x1_0 = self.conv1_0(self.hartley_pool_x1_0(x0_0))
        elif self.pooling == 'Hybrid':
            x1_0 = self.conv1_0(self.hybrid_pool_x1_0(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        
        if self.pooling == 'Max':
            x2_0 = self.conv2_0(self.pool(x1_0))
        elif self.pooling == 'Hartley':
            x2_0 = self.conv2_0(self.hartley_pool_x2_0(x1_0))
        elif self.pooling == 'Hybrid':
            x2_0 = self.conv2_0(self.hybrid_pool_x2_0(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        
        if self.pooling == 'Max':
            x3_0 = self.conv3_0(self.pool(x2_0))
        elif self.pooling == 'Hartley':
            x3_0 = self.conv3_0(self.hartley_pool_x3_0(x2_0))
        elif self.pooling == 'Hybrid':
            x3_0 = self.conv3_0(self.hybrid_pool_x3_0(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        
        if self.pooling == 'Max':
            x4_0 = self.conv4_0(self.pool(x3_0))
        elif self.pooling == 'Hartley':
            x4_0 = self.conv4_0(self.hartley_pool_x4_0(x3_0))
        elif self.pooling == 'Hybrid':
            x4_0 = self.conv4_0(self.hybrid_pool_x4_0(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output

    def init_weights(self):
        self.apply(weights_init)
        

if __name__ == '__main__':
    
    x = torch.rand([8, 1, 128, 128])
    unet = UnetSharp()
    a = unet(x)
    print(a.shape)
    