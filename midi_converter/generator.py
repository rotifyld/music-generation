from typing import Callable

import torch
from song import from_tensor

from midi_converter.converter import song_to_midi


def random_midis(decoder: Callable[[torch.Tensor], torch.Tensor], epoch, data_length, number=3,
                 thresholds=None, cuda=False):
    if thresholds is None:
        thresholds = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]

    fs = torch.normal(0.0, 1.0, [number, 128])
    if cuda:
        fs = fs.cuda()

    fs_decoded = [decoder(f) for f in fs]
    fs.cpu()

    for i, f in enumerate(fs_decoded):
        for th in thresholds:
            song = from_tensor(f, th)
            song_to_midi(song, 'export/midi/autoencoder/d{}_e{}_n{}_th{}'
                         .format(data_length, epoch, i // len(thresholds), th))
