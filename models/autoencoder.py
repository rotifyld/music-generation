import torch
from torch import nn

from song import ATOMS_IN_MEASURE, MEASURES_IN_SONG, NUM_PITCHES


class Autoencoder(nn.Module):
    def __init__(self, measure_in_features, measure_out_features, num_measures, song_out_features, cuda=False):
        super(Autoencoder, self).__init__()

        self.num_measures = num_measures
        self.CUDA = cuda

        self.measureEncoder = nn.Sequential(
            nn.Linear(measure_in_features, 1024),  # 4224 -> 1024
            nn.ReLU(True),
            nn.Linear(1024, measure_out_features),  # 1024 -> 128
            nn.ReLU(True)
        )

        self.songEncoder = nn.Sequential(
            nn.Linear(measure_out_features * num_measures, 512),  # 2048 -> 512
            nn.ReLU(True),
            nn.Linear(512, song_out_features),  # 512 -> 128
            nn.BatchNorm1d(128)
        )

        self.songDecoder = nn.Sequential(
            nn.Linear(song_out_features, 512),  # 512 <- 128
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, measure_out_features * num_measures),  # 2048 <- 512
        )

        self.measureDecoder = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(measure_out_features, 1024),  # 1024 <- 128
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, measure_in_features),  # 4224 <- 1024
            nn.Sigmoid()
        )

    def forward(self, x):  # x is of size D * num_H

        shape = x.shape

        x = torch.reshape(x, (shape[0] * self.num_measures, -1))

        x = self.measureEncoder(x)

        x = torch.reshape(x, (shape[0], -1))

        x = self.songEncoder(x)
        x = self.songDecoder(x)

        x = torch.reshape(x, (shape[0] * self.num_measures, -1))

        x = self.measureDecoder(x)

        x = torch.reshape(x, (shape[0], -1))

        return x

    def decode(self, feature_vector):  # feat_vector is tensor of length F

        x = torch.reshape(feature_vector, [1, -1])

        x = self.songDecoder(x)

        x = torch.reshape(x, [self.num_measures, -1])

        x = self.measureDecoder(x)
        return x


def build_model(cuda: bool) -> Autoencoder:
    model = Autoencoder(NUM_PITCHES * ATOMS_IN_MEASURE, 128, MEASURES_IN_SONG, 128, cuda)

    return model


def load_model(path: str) -> Autoencoder:
    model = Autoencoder(NUM_PITCHES * ATOMS_IN_MEASURE, 128, MEASURES_IN_SONG, 128, False)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model
