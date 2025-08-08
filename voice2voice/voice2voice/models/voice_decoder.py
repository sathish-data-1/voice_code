# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class VoiceDecoder(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         decoder_channels = config['model']['decoder_channels']
#         decoder_kernel_sizes = config['model']['decoder_kernel_sizes']
#         decoder_strides = config['model']['decoder_strides']
        
#         self.res_blocks = nn.Sequential(
#             *[ResBlock(decoder_channels[0]) for _ in range(3)]
#         )
        
#         self.deconvs = nn.ModuleList()
#         in_channels = decoder_channels[0]
#         for out_channels, kernel_size, stride in zip(
#             decoder_channels[1:], decoder_kernel_sizes[1:], decoder_strides[1:]):
#             self.deconvs.append(
#                 nn.ConvTranspose1d(
#                     in_channels, out_channels,
#                     kernel_size, stride,
#                     padding=(kernel_size-stride)//2,
#                     output_padding=stride-1))
#             in_channels = out_channels
            
#     def forward(self, x):
#         x = self.res_blocks(x)
#         for deconv in self.deconvs[:-1]:
#             x = F.leaky_relu(deconv(x), 0.1)
#         x = torch.tanh(self.deconvs[-1](x))
#         return x

import torch
import torch.nn as nn
import torch.nn.functional as F
from .voice_encoder import ResBlock

class VoiceDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        decoder_channels = config['model']['decoder_channels']
        decoder_kernel_sizes = config['model']['decoder_kernel_sizes']
        decoder_strides = config['model']['decoder_strides']
        
        self.res_blocks = nn.Sequential(
            *[ResBlock(decoder_channels[0]) for _ in range(3)]
        )
        
        self.deconvs = nn.ModuleList()
        in_channels = decoder_channels[0]
        for out_channels, kernel_size, stride in zip(
            decoder_channels[1:], decoder_kernel_sizes[1:], decoder_strides[1:]):
            self.deconvs.append(
                nn.ConvTranspose1d(
                    in_channels, out_channels,
                    kernel_size, stride,
                    padding=(kernel_size-stride)//2,
                    output_padding=stride-1))
            in_channels = out_channels
            
    def forward(self, x):
        x = self.res_blocks(x)
        for deconv in self.deconvs[:-1]:
            x = F.leaky_relu(deconv(x), 0.1)
        x = torch.tanh(self.deconvs[-1](x))
        return x