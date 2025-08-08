# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class ResBlock(nn.Module):
#     def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
#         super().__init__()
#         self.convs1 = nn.ModuleList([
#             nn.Conv1d(channels, channels, kernel_size, 
#                      padding='same', dilation=d)
#             for d in dilation
#         ])
#         self.convs2 = nn.ModuleList([
#             nn.Conv1d(channels, channels, kernel_size, 
#                      padding='same', dilation=1)
#             for _ in dilation
#         ])
        
#     def forward(self, x):
#         for c1, c2 in zip(self.convs1, self.convs2):
#             xt = F.leaky_relu(x, 0.1)
#             xt = c1(xt)
#             xt = F.leaky_relu(xt, 0.1)
#             xt = c2(xt)
#             x = xt + x
#         return x

# class VoiceEncoder(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         encoder_channels = config['model']['encoder_channels']
#         encoder_kernel_sizes = config['model']['encoder_kernel_sizes']
#         encoder_strides = config['model']['encoder_strides']
        
#         self.convs = nn.ModuleList()
#         in_channels = 1
#         for out_channels, kernel_size, stride in zip(
#             encoder_channels, encoder_kernel_sizes, encoder_strides):
#             self.convs.append(
#                 nn.Conv1d(in_channels, out_channels, 
#                           kernel_size, stride, 
#                           padding=(kernel_size-1)//2))
#             in_channels = out_channels
            
#         self.res_blocks = nn.Sequential(
#             *[ResBlock(in_channels) for _ in range(3)]
#         )
        
#     def forward(self, x):
#         for conv in self.convs:
#             x = F.leaky_relu(conv(x), 0.1)
#         x = self.res_blocks(x)
#         return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 
                     padding='same', dilation=d)
            for d in dilation
        ])
        self.convs2 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 
                     padding='same', dilation=1)
            for _ in dilation
        ])
        
    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x

class VoiceEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        encoder_channels = config['model']['encoder_channels']
        encoder_kernel_sizes = config['model']['encoder_kernel_sizes']
        encoder_strides = config['model']['encoder_strides']
        
        self.convs = nn.ModuleList()
        in_channels = 1
        for out_channels, kernel_size, stride in zip(
            encoder_channels, encoder_kernel_sizes, encoder_strides):
            self.convs.append(
                nn.Conv1d(in_channels, out_channels, 
                          kernel_size, stride, 
                          padding=(kernel_size-1)//2))
            in_channels = out_channels
            
        self.res_blocks = nn.Sequential(
            *[ResBlock(in_channels) for _ in range(3)]
        )
        
    def forward(self, x):
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
        x = self.res_blocks(x)
        return x