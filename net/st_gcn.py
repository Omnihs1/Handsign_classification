import torch.nn as nn
import torch
import torch.nn.functional as F
from gcn import GCN
from tcn import TCN
from graph import Graph, link
from torchinfo import summary

class ST_GCN(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride = 1, 
                 dropout = 0, 
                 residual = True):
        
        super(ST_GCN, self).__init__()
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride = (stride, 1)),
                nn.BatchNorm2d(out_channels)
            )

        self.spatial_block = GCN(in_channels, out_channels, kernel_size[1])
        self.temporal_block = TCN(out_channels, kernel_size[0], stride = stride)
        self.relu = nn.ReLU()
    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.spatial_block(x, A)
        # print(3, x.shape)
        x = self.temporal_block(x) + res
        # print(4, x.shape)
        return self.relu(x), A

class Model(nn.Module):
    def __init__(self, 
                in_channels, 
                spatial_kernel_size, 
                temporal_kernel_size,
                num_class, 
                graph_args):
        super().__init__()

        #Load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A2, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = spatial_kernel_size
        temporal_kernel_size = temporal_kernel_size
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        self.st_gcn_networks = nn.ModuleList((
            ST_GCN(in_channels, 64, kernel_size, 1, residual=False),
            ST_GCN(64, 64, kernel_size, 1),
            ST_GCN(64, 64, kernel_size, 1),
            ST_GCN(64, 64, kernel_size, 1),
            ST_GCN(64, 128, kernel_size, 2),
            ST_GCN(128, 128, kernel_size, 1),
            ST_GCN(128, 128, kernel_size, 1),
            ST_GCN(128, 256, kernel_size, 2),
            ST_GCN(256, 256, kernel_size, 1),
            ST_GCN(256, 256, kernel_size, 1),
        ))

        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, x):

        # data normalization
        N, C, T, V= x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, C, T, V)
        # forwad
        for gcn in self.st_gcn_networks:
            x, _ = gcn(x, self.A)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        # print(5, x.shape)
        # x = x.view(N, -1).mean(dim=1)
        # print(6, x.shape)
        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)

        return x
    
if __name__ == '__main__':
    a = Model(in_channels = 3, 
              spatial_kernel_size = 1, 
              temporal_kernel_size = 9,
              num_class = 50, 
              graph_args={"link":link, "node":25})
    # x = torch.randn((32, 3, 80, 25))
    summary(a, input_size=(32, 3, 80, 25), col_names=["input_size", "output_size", "num_params"])
    # a(x)
    # print(a.parameters())