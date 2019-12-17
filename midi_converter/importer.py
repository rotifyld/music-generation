import os
import traceback
from typing import Tuple, List

import torch
from mido import MidiFile

from logger import *
from song import Song
from midi_converter.converter import midi_to_song


def get_data(name='ninsheetmusic.pt') -> torch.Tensor:
    songs = torch.load('/home/rotifyld/Code/Python/jnp/music_generation/midi_converter/ninsheetmusic.pt')
    return songs


def import_all_midis(path: str) -> List[Song]:
    songs = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith('.mid') or file.endswith('.midi'):
                try:
                    log_info('Trying to open \"{}\".'.format(file))
                    midi = MidiFile(os.path.join(root, file))
                    songs.append(midi_to_song(midi))
                    log_ok('Converted correctly.')
                except Exception as e:
                    log_error('Exception during converting \"{}\": {}.\n'.format(file, str(e)))
                    traceback.print_exc()
    return songs

