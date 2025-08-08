# import torch
# import torch.nn as nn
# import math

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super().__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)
        
#     def forward(self, x):
#         return x + self.pe[:, :x.size(1)]

# class SemanticTransformer(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         model_config = config['model']
        
#         self.input_proj = nn.Conv1d(
#             model_config['encoder_channels'][-1],
#             model_config['embedding_dim'],
#             1)
        
#         self.pos_encoder = PositionalEncoding(model_config['embedding_dim'])
        
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=model_config['embedding_dim'],
#             nhead=model_config['transformer_heads'],
#             dim_feedforward=model_config['transformer_ff_dim'],
#             batch_first=True)
        
#         self.transformer = nn.TransformerEncoder(
#             encoder_layer,
#             num_layers=model_config['transformer_layers'])
        
#     def forward(self, x):
#         # x: [batch, channels, time]
#         x = self.input_proj(x)
#         x = x.permute(0, 2, 1)  # [batch, time, channels]
#         x = self.pos_encoder(x)
#         x = self.transformer(x)
#         x = x.permute(0, 1, 2)  # [batch, channels, time]
#         return x


import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class SemanticTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        model_config = config['model']
        
        self.input_proj = nn.Conv1d(
            model_config['encoder_channels'][-1],
            model_config['embedding_dim'],
            1)
        
        self.pos_encoder = PositionalEncoding(model_config['embedding_dim'])
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_config['embedding_dim'],
            nhead=model_config['transformer_heads'],
            dim_feedforward=model_config['transformer_ff_dim'],
            batch_first=True)
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=model_config['transformer_layers'])
        
    def forward(self, x):
        x = self.input_proj(x)
        x = x.permute(0, 2, 1)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.permute(0, 1, 2)
        return x