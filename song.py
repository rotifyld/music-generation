from typing import List, Union

import torch

import util

MEASURES_IN_SONG = 16  # def 16
ATOMS_IN_MEASURE = 48  # def 48
ATOMS_IN_SONG = MEASURES_IN_SONG * ATOMS_IN_MEASURE

MIN_NOTE = 21
MAX_NOTE = 108
NUM_NOTES = MAX_NOTE - MIN_NOTE + 1

_key_str_to_int = {
    'Gb': -6,
    'G': -5,
    'G#': -4,

    'Ab': -4,
    'A': -3,
    'A#': -2,

    'Bb': -2,
    'B': -1,
    'B#': 0,

    'Cb': -1,
    'C': 0,
    'C#': 1,

    'Db': 1,
    'D': 2,
    'D#': 3,

    'Eb': 3,
    'E': 4,
    'E#': 5,

    'Fb': 4,
    'F': 5,
    'F#': 6,
}


def _key_str_to_transpose(key: str):
    if key[-1] == 'm':
        key = key[:-1]
    return - _key_str_to_int.get(key, 0)


class Song:
    data: Union[List[List[List[int]]], torch.Tensor]

    def __init__(self, key='C'):
        self.key = key
        self.transposition = -1 * _key_str_to_transpose(key)
        self.data = [[[0 for _ in range(NUM_NOTES)] for _ in range(ATOMS_IN_MEASURE)] for _ in
                     range(MEASURES_IN_SONG)]

    def visualize(self) -> str:
        NOTE = '█'
        BREAK = '·'
        return util.transpose_and_flip_upside_down_str(MEASURES_IN_SONG, ATOMS_IN_MEASURE,
                                                       '\n'.join(
                                                           '\n'.join(
                                                               ''.join(NOTE if item else BREAK for item in row)
                                                               for row in measure)
                                                           for measure in self.data))

    def _add_note(self, atom: int, note: int) -> bool:
        if atom >= ATOMS_IN_SONG:
            return False
        else:
            self.data[atom // ATOMS_IN_MEASURE][atom % ATOMS_IN_MEASURE][note + self.transposition - MIN_NOTE] = 1
            return True

    def add_note(self, tick: int, ticks_per_measure: int, note: int) -> bool:
        return self._add_note(int(round(ATOMS_IN_MEASURE * tick / ticks_per_measure)), note)  # round > floor

    def is_correct_song(self) -> bool:
        # return any(any(note for note in bar) for bar in self.data[ATOMS_IN_SONG - ATOMS_IN_MEASURE:ATOMS_IN_SONG])
        raise DeprecationWarning

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor(self.data).flatten()


ZEROS = torch.zeros([16, 48, 88])
ONES = torch.ones([16, 48, 88])
if (torch.cuda.is_available()):
    ZEROS = ZEROS.cuda()
    ONES = ONES.cuda()


def from_tensor(t: torch.Tensor, threshold=0.5) -> Song:
    s = Song()
    t = t.reshape([16, 48, 88])
    t = torch.where(t > threshold, ONES, ZEROS)
    s.data = t
    return s
