import os
from pprint import pprint
from typing import List, Iterable

from logger import log_info
from song import Song
from importer import import_all_midis


songs = import_all_midis()
for song in songs:
    if song.key == 'C':
        log_info('UUU')
        print(song.visualize())