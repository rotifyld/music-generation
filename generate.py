import torch
from torch.autograd import Variable

from midi_converter.importer import get_data
from midi_converter.generator import random_midis
from logger import log_info
from models.autoencoder import load_model

MODEL = "modelE200D10T2019-12-17 19:27:26.989712.pt"


def generate():
    model = load_model(MODEL)
    random_midis(model.decode, epoch=200, data_length=10, thresholds=[0.1, 0.06, 0.03], number=20, cuda=False)


def encode():
    def prepare_data(data):
        data = torch.flatten(data)
        data = data.float()
        data = Variable(data)
        return data

    log_info('Loading data and model...')
    dataset = get_data()[:10]
    log_info('Data loaded!')
    model = load_model(MODEL)
    log_info('Model loaded!')

    data_encoded = [model.encode(prepare_data(data)) for data in dataset]
    data_encoded = data_encoded


if __name__ == '__main__':
    # generate()
    encode()
