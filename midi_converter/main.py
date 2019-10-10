from logger import log_info
from midi_converter.importer import import_all_midis


songs = import_all_midis()
for song in songs:
    if song.key == 'C':
        log_info('UUU')
        print(song.visualize())