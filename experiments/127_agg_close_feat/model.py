# Several SqueezeFormer components where copied/ adapted from https://github.com/upskyy/Squeezeformer/


import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchinfo import summary


class TransformerEncoderBlock(nn.Module):
    def __init__(self, n_dim, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()

        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(self, x):
        # x shape: (batch, 384, n_dim)
        x = self.transformer_encoder(x)
        return x


if __name__ == "__main__":
    batch_size = 4
    seq_len = 384
    n_dim = 64 * 60

    model = TransformerEncoderBlock(
        n_dim=n_dim, nhead=8, num_layers=2, dim_feedforward=2048
    )

    summary(model, input_size=(batch_size, seq_len, n_dim))
