from datetime import datetime

import torch
from torch import nn
from torch.autograd import Variable

from midi_converter.importer import get_data
from logger import log_info, log_ok
from models.autoencoder import Autoencoder, build_model

NUM_EPOCHS = 10000
SAVE_MODEL_EACH = 1000
SHOW_EXACT_LOSS_EACH = 9999999
DATASET = 'ninsheetmusic_trans'
DATA_LENGTH = 4016  # 4016 = 251 * 2^4 is the max
BATCH_SIZE = 251

MODEL = 'autoencoder_batch'
CUDA = True

# optimizer
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

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

    loss_global_history = []

    # train loop
    for epoch in range(NUM_EPOCHS):

        loss_local_history = []

        # prepare batches
        perm = torch.randperm(DATA_LENGTH)
        shuffled_data = dataset[perm]
        batched_data = torch.chunk(shuffled_data, DATA_LENGTH // BATCH_SIZE)

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

            loss_local_history.append(loss.data)

            # logging
            # if not CUDA:
            #     if i in [10, 25, 50] or i % 100 == 0:
            #         log_ok('Processed {}/{} songs.'.format(i + 1, dataset_length))
            # else:
            #     if i == 100 or i % 1000 == 0:
            #         log_ok('Processed {}/{} songs.'.format(i + 1, dataset_length))

        # after each epoch
        mean_loss = torch.tensor(loss_local_history).mean()
        loss_global_history.append(mean_loss)

        if epoch != 0 and epoch % SAVE_MODEL_EACH == 0:
            torch.save(model.state_dict(),
                       'export/models/{}/{}_d{}_e{}_l{:.16f}.pt'.format(MODEL, start_time, dataset_length, epoch,
                                                                        mean_loss))

        if epoch != 0 and epoch % SHOW_EXACT_LOSS_EACH == 0:
            log_info('loss: {}'.format(loss_local_history))

        # ===================log========================
        log_info('Epoch [{}/{}], loss:{:.16f}. Time elapsed: {}'
                 .format(epoch + 1, NUM_EPOCHS, mean_loss, datetime.now() - start_time))

    log_info('Finished training in {}'.format(datetime.now() - start_time))
    torch.save(model.state_dict(),
               'export/models/{}/{}_fin_d{}_e{}_l{:.16f}.pt'.format(MODEL, start_time, dataset_length, NUM_EPOCHS,
                                                                    mean_loss))
    log_ok('Model saved.')


if __name__ == '__main__':
    log_info('{} CUDA device(s) available.'.format(
        torch.cuda.device_count()) if torch.cuda.is_available() else 'No CUDA available.')

    if CUDA:
        torch.cuda.empty_cache()

    train()
