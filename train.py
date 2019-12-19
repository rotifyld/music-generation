from datetime import datetime

import torch
from torch import nn
from torch.autograd import Variable

from midi_converter.importer import get_data
from logger import log_info, log_ok
from models.autoencoder import Autoencoder, build_model

NUM_EPOCHS = 10

DATASET = 'ninsheetmusic'
DATA_LENGTH = -1  # -1 for all available in given dataset

MODEL = 'autoencoder'
CUDA = False

# optimizer
LEARNING_RATE = 1e-2
WEIGHT_DECAY = 1e-5


def train():
    log_info('Loading data...')
    dataset = get_data('ninsheetmusic')[:DATA_LENGTH]
    dataset_length = dataset.size()[0]
    log_info('Data loaded! Dataset of size {}'.format(dataset_length))

    log_info('Building model...')
    model: Autoencoder = build_model(CUDA)
    if CUDA:
        model = model.cuda()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    log_info('Model built! Starting training..')
    start_time = datetime.now()

    # train loop
    for epoch in range(NUM_EPOCHS):
        for i, data in enumerate(dataset):
            data = Variable(data)
            if CUDA:
                data = data.cuda()

            # forward
            output = model(data)
            loss = criterion(output, data)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # logging
            if not CUDA:
                if i in [10, 25, 50] or i % 100 == 0:
                    log_ok('Processed {}/{} songs.'.format(i + 1, dataset_length))
            else:
                if i == 100 or i % 1000 == 0:
                    log_ok('Processed {}/{} songs.'.format(i + 1, dataset_length))

        # ===================log========================
        log_info('Epoch [{}/{}], loss:{:.8f}. Time elapsed: {}'
                 .format(epoch + 1, NUM_EPOCHS, loss.data, datetime.now() - start_time))

    log_info('Finished training in {}'.format(datetime.now() - start_time))
    torch.save(model.state_dict(),
               'export/models/{}/{}_d{}_e{}.pt'.format(MODEL, datetime.now(), dataset_length, NUM_EPOCHS))
    log_ok('Model saved.')


if __name__ == '__main__':
    log_info('{} CUDA device(s) available.'.format(
        torch.cuda.device_count()) if torch.cuda.is_available() else 'No CUDA available.')

    if CUDA:
        torch.cuda.empty_cache()

    train()
