import os
from typing import List

import torch

from midi_converter.importer import import_all_midis, get_data
from song import Song


def convert_midi_to_tensor(name: str):
    if name is not None:
        path = '{}/../data_scraping/{}'.format(os.path.dirname(os.path.abspath(__file__)), name)
        songs = import_all_midis(path)
        songs_flattened = [s.to_tensor() for s in songs]
        songs_flattened = torch.stack(songs_flattened)
        torch.save(songs_flattened, '{}.pt'.format(name))


def edit_tensor(name):
    songs = get_data(name)
    songs = [torch.flatten(s) for s in songs]
    songs = torch.stack(songs)
    songs = songs.float()
    torch.save(songs, 'ninsheetmusic2.pt')


if __name__ == '__main__':
    convert_midi_to_tensor('ninsheetmusic1')
    # edit_tensor('ninsheetmusic')
