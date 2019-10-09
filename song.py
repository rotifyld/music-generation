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


class Song:
    data: List[List[int]]

    def __init__(self, key: int):
        self.key = key
        self.data = [[0 for _ in range(NUM_NOTES)] for _ in range(ATOMS_IN_SONG)]

    def visualize(self) -> str:
        NOTE = '█'
        BREAK = '·'
        return util.transpose_and_flip_upside_down_str('\n'.join(''.join(NOTE if item == 1 else BREAK for item in row) for row in self.data))

    def _add_note(self, atom: int, note: int) -> bool:
        if atom >= ATOMS_IN_SONG:
            return False
        else:
            self.data[atom][note - MIN_NOTE] = 1
            return True

    def add_note(self, tick: int, ticks_per_measure: int, note: int) -> bool:
        # log_info('adding note {:3d} tick {:7d}/{:f}'.format(note, tick, ticks_per_measure))
        return self._add_note(int(round(ATOMS_IN_MEASURE * tick / ticks_per_measure)), note)
