import torch.nn as nn

class SemanticEncoderTC(nn.Module):
    def __init__(self):
        super(SemanticEncoderTC, self).__init__()
        
        # No layers in the encoder
        
    def forward(self, x):
        # No encoding operations
        return x

class SemanticDecoderTC(nn.Module):
    def __init__(self):
        super(SemanticDecoderTC, self).__init__()
        
        # No layers in the decoder
        
    def forward(self, x):
        # No decoding operations
        return x
