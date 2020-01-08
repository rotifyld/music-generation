from datetime import datetime

import torch
from torch import nn
from torch.autograd import Variable

from midi_converter.importer import get_data
from logger import log_info, log_ok
from models.autoencoder import Autoencoder, build_model

NUM_EPOCHS = 2000
SAVE_MODEL_EACH = 200
DATASET = 'ninsheetmusic'
DATA_LENGTH = 10  # 4016 = 251 * 2^4 is the max

MODEL = 'autoencoder_batch'
CUDA = True

# optimizer
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 2

assert (DATA_LENGTH % BATCH_SIZE == 0)


def train():
    log_info('Loading data...')
    dataset = get_data('ninsheetmusic')[:DATA_LENGTH]
    dataset_length = dataset.size()[0]
    log_info('Data loaded! Dataset of size {}'.format(dataset_length))

    log_info('Building model...')
    model: Autoencoder = build_model(CUDA)
    if CUDA:
        model = model.cuda()

    criterion = nn.BCELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE)

    log_info('Model built! Starting training...')
    start_time = datetime.now()

    # prepare batches
    batched_data = torch.chunk(dataset, DATA_LENGTH // BATCH_SIZE)

    # train loop
    for epoch in range(NUM_EPOCHS):
        for i, data in enumerate(batched_data):
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
            # if not CUDA:
            #     if i in [10, 25, 50] or i % 100 == 0:
            #         log_ok('Processed {}/{} songs.'.format(i + 1, dataset_length))
            # else:
            #     if i == 100 or i % 1000 == 0:
            #         log_ok('Processed {}/{} songs.'.format(i + 1, dataset_length))

            if epoch % SAVE_MODEL_EACH == 0:
                torch.save(model.state_dict(),
                           'export/models/{}/{}_d{}_e{}_l{:.8f}.pt'.format(MODEL, start_time, dataset_length, epoch,
                                                                           loss.data))

        # ===================log========================
        log_info('Epoch [{}/{}], loss:{:.8f}. Time elapsed: {}'
                 .format(epoch + 1, NUM_EPOCHS, loss.data, datetime.now() - start_time))

    log_info('Finished training in {}'.format(datetime.now() - start_time))
    torch.save(model.state_dict(),
               'export/models/{}/{}_fin_d{}_e{}_l{:.8f}.pt'.format(MODEL, start_time, dataset_length, NUM_EPOCHS, loss.data))
    log_ok('Model saved.')


if __name__ == '__main__':
    log_info('{} CUDA device(s) available.'.format(
        torch.cuda.device_count()) if torch.cuda.is_available() else 'No CUDA available.')

    if CUDA:
        torch.cuda.empty_cache()

    train()
