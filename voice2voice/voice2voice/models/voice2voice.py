# from .voice_encoder import VoiceEncoder
# from .semantic_transformer import SemanticTransformer
# from .voice_decoder import VoiceDecoder
# import torch.nn as nn

# class Voice2VoiceModel(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.encoder = VoiceEncoder(config)
#         self.transformer = SemanticTransformer(config)
#         self.decoder = VoiceDecoder(config)
        
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.transformer(x)
#         x = self.decoder(x)
#         return x
    
#     def encode(self, x):
#         return self.encoder(x)
    
#     def transform(self, x):
#         return self.transformer(x)
    
#     def decode(self, x):
#         return self.decoder(x)


import torch.nn as nn
from .voice_encoder import VoiceEncoder
from .semantic_transformer import SemanticTransformer
from .voice_decoder import VoiceDecoder

class Voice2VoiceModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = VoiceEncoder(config)
        self.transformer = SemanticTransformer(config)
        self.decoder = VoiceDecoder(config)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.transformer(x)
        x = self.decoder(x)
        return x
    
    def encode(self, x):
        return self.encoder(x)
    
    def transform(self, x):
        return self.transformer(x)
    
    def decode(self, x):
        return self.decoder(x)