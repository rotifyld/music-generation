import os
from typing import List

import torch

from midi_converter.importer import import_all_midis, get_data
from song import Song, from_tensor, MIN_PITCH


def convert_midi_to_tensor(dir_name: str, dataset_name: str):
    if dir_name is not None:
        path = '{}/../data_scraping/{}'.format(os.path.dirname(os.path.abspath(__file__)), dir_name)
        songs = import_all_midis(path)
        songs = [s.to_tensor() for s in songs]
        songs = torch.stack(songs)
        songs = songs.float()

        torch.save(songs, '{}.pt'.format(dataset_name))


def edit_tensor(name):
    songs = get_data(name)
    songs = [torch.flatten(s) for s in songs]
    songs = torch.stack(songs)
    songs = songs.float()
    torch.save(songs, 'ninsheetmusic2.pt')


def peek_tensor(name):
    songs = get_data(name)
    cumulative_pitches = torch.zeros([13])
    for i, song in enumerate(songs):
        if i % 100 == 0:
            print('   C   C#  D   D#  E   F   F#  G   G#  A   A#  B   ')
        song = from_tensor(song, 0.5)
        pitches = torch.where(song.data)[2]
        pitches = pitches + MIN_PITCH - 60
        pitches = pitches % 12
        num_pitches = torch.cat((pitches, torch.tensor([12]))).bincount()
        cumulative_pitches += num_pitches

    pass
    total = cumulative_pitches.sum()
    precentage_pitches = cumulative_pitches / total

    # for i, song in enumerate(random_songs):
    #     song = from_tensor(song, 0.5)
    #     from midi_converter.converter import song_to_midi
    #     path = '{}/../export/midi/test/{}'.format(os.path.dirname(os.path.abspath(__file__)), i)
    #     song_to_midi(song, path)


if __name__ == '__main__':
    # convert_midi_to_tensor('poo', 'poo')
    # edit_tensor('')
    peek_tensor('ninsheetmusic_trans')
