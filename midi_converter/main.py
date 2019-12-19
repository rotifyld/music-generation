import os
from typing import List

import torch

from midi_converter.importer import import_all_midis, get_data
from song import Song


def convert_midi_to_tensor(name: str):
    if name is not None:
        songs = import_all_midis('{}/../data_scraping/{}'.format(os.path.dirname(os.path.abspath(__file__)), name))
        songs = [s.to_tensor() for s in songs]
        songs = torch.stack(songs)
        torch.save(songs, '{}.pt'.format(name))


def edit_tensor(name):
    songs = get_data(name)
    songs = [torch.flatten(s) for s in songs]
    songs = torch.stack(songs)
    songs = songs.float()
    torch.save(songs, 'ninsheetmusic2.pt')


if __name__ == '__main__':
    pass
    # convert_midi_to_tensor('ninsheetmusic')
    # edit_tensor('ninsheetmusic')
