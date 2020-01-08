import torch
from torch.autograd import Variable

from midi_converter.importer import get_data
from midi_converter.generator import random_midis
from logger import log_info
from models.autoencoder import load_model

from song import from_tensor
from midi_converter.converter import song_to_midi

MODEL = "export/models/autoencoder_batch/2020-01-08 16:09:55.178383_d10_e500.pt"


def generate():
    """Loads trained model and generates new melody from random midis"""

    model = load_model(MODEL)
    random_midis(model.decode, epoch=500, data_length=10, thresholds=[0.1, 0.06], number=10, cuda=False)


def generate_mean_song():
    model = load_model(MODEL)
    data_encoded = torch.load('data.pt')
    data_encoded = data_encoded

    # generate mean song
    mean_feature = torch.mean(torch.stack(data_encoded), 0)
    mean_decoded = model.decode(mean_feature)
    th = 0.99999997001
    mean_song = from_tensor(mean_decoded, th)
    song_to_midi(mean_song, 'export/midi/autoencoder_batch/d4016_e50_th{}'.format(th))


def encode():
    def prepare_data(data):
        data = torch.flatten(data)
        data = data.float()
        data = Variable(data)
        return data

    log_info('Loading data and model...')
    dataset = get_data()
    log_info('Data loaded!')
    model = load_model(MODEL)
    log_info('Model loaded!')

    data_encoded = [model.encode(prepare_data(data)) for data in dataset]

    torch.save(data_encoded, 'data.pt')


if __name__ == '__main__':
    generate()
    # generate_mean_song()
    # encode()
