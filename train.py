from datetime import datetime

import torch
from torch import nn
from torch.autograd import Variable

from midi_converter.converter import song_to_midi
from midi_converter.importer import get_data
from midi_converter.generator import random_midis
from logger import log_info, log_ok
from model import Autoencoder, build_model, load_model
from song import from_tensor

NUM_EPOCHS = 50
VERBOSE = True
CUDA = True


def train():
    dataset = get_data()[:50]
    print(dataset.size())
    dataset_length = dataset.size()[0]
    log_info('Dataset of size {}'.format(dataset_length))

    log_info('Building model...')
    model: Autoencoder = build_model(CUDA)
    if CUDA:
        model = model.cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    log_info('Started training.')
    start_time = datetime.now()
    for epoch in range(NUM_EPOCHS):
        for i, data in enumerate(dataset):
            data = torch.flatten(data)  # TODO data needs to be reloaded and then this line deleted
            data = data.float()
            data = Variable(data)
            if CUDA:
                data = data.cuda()
            # ===================forward=====================
            output = model(data)
            loss = criterion(output, data)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if VERBOSE and not CUDA:
                if i in [10, 25, 50] or i % 100 == 0:
                    log_ok('Processed {}/{} songs.'.format(i + 1, dataset_length))
            else:
                if i == 100 or i % 1000 == 0:
                    log_ok('Processed {}/{} songs.'.format(i + 1, dataset_length))

        # ===================log========================
        log_info('Epoch [{}/{}], loss:{:.4f}. Time elapsed: {}'
                 .format(epoch + 1, NUM_EPOCHS, loss.data, datetime.now() - start_time))

    log_info('Finished training in {}'.format(datetime.now() - start_time))
    torch.save(model.state_dict(), 'modelE{}D{}T{}.pt'.format(NUM_EPOCHS, dataset_length, datetime.now()))

    random_midis(model.decode, NUM_EPOCHS, dataset_length, cuda=CUDA)


def debug():
    dataset = get_data()[:10]
    for i, data in enumerate(dataset):
        data = torch.flatten(data)
        song = from_tensor(data, 0.5)
        song_to_midi(song, 'training{}'.format(i))


if __name__ == '__main__':
    log_info('{} CUDA device(s) available.'.format(torch.cuda.device_count()) if torch.cuda.is_available() else 'No CUDA available.')

    if CUDA:
        torch.cuda.empty_cache()

    train()
