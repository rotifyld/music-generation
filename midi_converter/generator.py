from typing import Callable

import torch
from song import from_tensor

from midi_converter.converter import midi_to_song, song_to_midi


def random_midis(decoder: Callable[[torch.Tensor], torch.Tensor], epoch, data_length, number=20, thresholds=None, cuda=False):
    if thresholds is None:
        thresholds = [3e-3, 1e-3, 3e-4]

    fs = torch.normal(0.0, 1.0, [number, 120])
    if cuda:
        fs = fs.cuda()

    fs_decoded = [decoder(f) for f in fs]
    fs.cpu()

    songs = [from_tensor(f_decoded, th) for f_decoded in fs_decoded for th in thresholds]

    midis = [song_to_midi(song, 'export/DL{}E{}N{}'.format(data_length, epoch, i)) for i, song in enumerate(songs)]


