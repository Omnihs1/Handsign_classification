import torch.nn as nn
from torchinfo import summary

class TCN(nn.Module):
    def __init__(self,  
                out_channels, 
                kernel_size, 
                stride = 1,
                dropout=0):
        
        super(TCN, self).__init__()
        padding = ((kernel_size - 1) // 2, 0)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size, 1),
                stride = (stride, 1),
                padding= padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        x = self.tcn(x)
        return x
    

if __name__ == '__main__':
    a = TCN(64, 9, 2)
    summary(a, input_size=(32, 64, 80, 25), col_names=["input_size", "output_size", "num_params"])