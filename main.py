import os
from pprint import pprint
from typing import List, Iterable

from song import Song
from importer import import_all_midis


songs = import_all_midis()
for song in songs:
    print(song.visualize())
