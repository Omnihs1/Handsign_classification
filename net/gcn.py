import torch.nn as nn
import torch
from graph import Graph, link
from torchinfo import summary

class GCN(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 t_kernel_size=1,
                 t_stride = 1,
                 t_padding = 0,
                 t_dilation = 1):
        super(GCN, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size = (t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1)
        )

    def forward(self, x, A):
        """
            Input shape: (N, in_channels, V, T)
            N : batch size
            in_channels : numbers of features in
            V : numbers of joints (as nodes)
            T : numbers of frames
            Output shape: (N, out_channels, V, T)
            N : batch size
            out_channels : numbers of features out
            V : numbers of joints (as nodes)
            T : numbers of frames
        """

        x = self.conv2d(x)
        # Only change out_channels
        N, C, T, V = x.size()
        # print(1, x.size())
        # print(2, A.shape)
        x = torch.einsum("nctv,vw -> nctw", (x, A))
        return x.contiguous(), A

if __name__ == "__main__":
    a = Graph(link = link, node = 25)
    b = GCN(2, 15)
    x = torch.randn(size = (32, 2, 80, 25))
    A2 = torch.tensor(a.A2, dtype=torch.float32, requires_grad=False)
    y, A3 = b(x, A2)
    print(y.shape)
    summary(b)
    # print(A3)
    # print(A2)
