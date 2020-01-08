import torch
from torch import nn

from models.time_distributed import TimeDistributed
from song import ATOMS_IN_MEASURE, MEASURES_IN_SONG, NUM_PITCHES


class Autoencoder(nn.Module):
    def __init__(self, chunk_in_features, chunk_out_features, num_chunks, out_features, cuda=False):
        super(Autoencoder, self).__init__()

        self.num_chunks = num_chunks
        self.CUDA = cuda

        self.encoderChunk = nn.Sequential(
            nn.Linear(chunk_in_features, 1024),  # 4224 -> 1024
            nn.ReLU(True),
            nn.Linear(1024, chunk_out_features),  # 1024 -> 128
            nn.ReLU(True)
        )

        self.encoder = nn.Sequential(
            nn.Linear(chunk_out_features * num_chunks, 512),  # 2048 -> 512
            nn.ReLU(True),
            nn.Linear(512, out_features),  # 512 -> 128
            nn.BatchNorm1d(128)
        )

        self.decoder = nn.Sequential(
            nn.Linear(out_features, 512),  # 512 <- 128
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, chunk_out_features * num_chunks),  # 2048 <- 512
        )

        self.decoderChunk = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(chunk_out_features, 1024),  # 1024 <- 128
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, chunk_in_features),  # 4224 <- 1024
            nn.Sigmoid()
        )

    def forward(self, x):  # x is of size D * num_H

        shape = x.shape

        x = torch.reshape(x, (shape[0] * self.num_chunks, -1))

        x = self.encoderChunk(x)

        x = torch.reshape(x, (shape[0], -1))

        x = self.encoder(x)
        x = self.decoder(x)

        x = torch.reshape(x, (shape[0] * self.num_chunks, -1))

        x = self.decoderChunk(x)

        x = torch.reshape(x, (shape[0], -1))

        return x

    def decode(self, feature_vector):  # feat_vector is tensor of length F

        x = torch.reshape(feature_vector, [1, -1])

        x = self.decoder(x)

        x = torch.reshape(x, [self.num_chunks, -1])

        x = self.decoderChunk(x)
        return x

    def encode(self, x):
        x_chunks = torch.chunk(x, self.num_chunks)
        x_chunks = [self.encoderChunk(chunk) for chunk in x_chunks]
        x = torch.cat(x_chunks)

        x = self.encoder(x)
        return x


def build_model(cuda: bool) -> Autoencoder:
    model = Autoencoder(NUM_PITCHES * ATOMS_IN_MEASURE, 128, MEASURES_IN_SONG, 128, cuda)

    return model


def load_model(path: str) -> Autoencoder:
    model = Autoencoder(NUM_PITCHES * ATOMS_IN_MEASURE, 128, MEASURES_IN_SONG, 128, False)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model
