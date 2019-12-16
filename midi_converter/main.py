import torch

from logger import log_info
from midi_converter.importer import import_all_midis, song_to_midi

# songs = import_all_midis('midi')
songs = import_all_midis('../data_scraping/ninsheetmusic')
songs = list(map(lambda s: s.data, songs))
songs = torch.stack(songs)

torch.save(songs, 'ninsheetmusic.pt')


# i = 0
# for song in songs:
#     song_to_midi(song, '{}.mid'.format(i))
#     i += 1
