import torch
from torch import nn

from song import ATOMS_IN_MEASURE, MEASURES_IN_SONG, NUM_NOTES


class Autoencoder(nn.Module):
    def __init__(self, D, H, num_chunks, F, CUDA=False):
        super(Autoencoder, self).__init__()

        self.D = D
        self.H = H
        self.num_chunks = num_chunks
        self.F = F
        self.CUDA = CUDA

        self.encoderChunk = nn.Sequential(
            nn.Linear(D, H),  # D = 4224
            nn.ReLU(True)
        )

        self.encoder = nn.Sequential(
            nn.Linear(H * num_chunks, F),  # H * num_H = 3200
            nn.Tanh()
        )

        self.decoder = nn.Sequential(
            nn.Linear(F, H * num_chunks),
            nn.ReLU(True)
        )

        self.decoderChunk = nn.Sequential(
            nn.Linear(H, D),
            nn.Tanh()
        )

    def forward(self, x):  # x is of size D * num_H

        x_chunks = torch.chunk(x, self.num_chunks)  # split input into 16 chunks of length D
        x_chunks = [self.encoderChunk(chunk) for chunk in x_chunks]
        x = torch.cat(x_chunks)

        x = self.encoder(x)
        x = self.decoder(x)

        x_chunks = torch.chunk(x, self.num_chunks)
        x_chunks = [self.decoderChunk(chunk) for chunk in x_chunks]
        x = torch.cat(x_chunks)

        return x

    def decode(self, feature_vector):  # feat_vector is tensor of length F
        x = self.decoder(feature_vector)

        x_chunks = torch.chunk(x, self.num_chunks)
        x_chunks = [self.decoderChunk(chunk) for chunk in x_chunks]
        x = torch.cat(x_chunks)

        return x


def build_model(cuda: bool) -> Autoencoder:
    model = Autoencoder(NUM_NOTES * ATOMS_IN_MEASURE, 200, MEASURES_IN_SONG, 120, cuda)

    return model


def load_model() -> Autoencoder:
    raise NotImplementedError
