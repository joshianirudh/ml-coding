import torch
from torch import nn

class Encoder(nn.module):
    def __init__(self):
        self.mha = MHA()
        self.norm = Norm()
        self.ffn = FFN()
        self.norm2 = Norm()
    def forward(self):
        pass

class Decoder(nn.module):
    def __init__(self):
        self.m_mha = MHA(mask=True)
        self.norm = Norm()
        self.ffn = FFN()
        self.norm2 = Norm()

    def forward(self):
        pass


class CrossAttentionDecoder(nn.module):
    def __init__(self):
        self.m_mha = MHA(causal_mask = True)
        self.norm = Norm()
        self.mha = MHA()
        self.norm2 = Norm()
        self.ffn = FFN()
        self.norm3 = Norm()
    def forward(self):
        pass



class LLM(nn.module):
    def __init__(self, n_encoders, n_decoders):
        pass
