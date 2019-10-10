import math
from typing import List

import util
from logger import log_info

MEASURES_IN_SONG = 16
ATOMS_IN_MEASURE = 48
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


def key_str_to_transpose(key: str):
    if key[-1] == 'm':
        key = key[:-1]
    return - _key_str_to_int.get(key, 0)


class Song:
    data: List[List[int]]

    def __init__(self, key: str):
        self.key = key
        self.transposition = -1 * key_str_to_transpose(key)
        self.data = [[0 for _ in range(NUM_NOTES)] for _ in range(ATOMS_IN_SONG)]

    def visualize(self) -> str:
        NOTE = '█'
        BREAK = '·'
        return util.transpose_and_flip_upside_down_str(
            '\n'.join(''.join(NOTE if item == 1 else BREAK for item in row) for row in self.data))

    def _add_note(self, atom: int, note: int) -> bool:
        if atom >= ATOMS_IN_SONG:
            return False
        else:
            self.data[atom][note + self.transposition - MIN_NOTE] = 1
            return True

    def add_note(self, tick: int, ticks_per_measure: int, note: int) -> bool:
        return self._add_note(int(round(ATOMS_IN_MEASURE * tick / ticks_per_measure)), note)

    def is_correct_song(self) -> bool:
        return any(any(note == 1 for note in bar) for bar in self.data[ATOMS_IN_SONG - ATOMS_IN_MEASURE:ATOMS_IN_SONG])
