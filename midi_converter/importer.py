import os
from typing import Tuple, List
from mido import MidiFile, MidiTrack, Message

from song import Song, SongFun
from logger import log_ok, log_info, log_error, log_warning


def import_all_midis() -> List[SongFun]:
    songs = []
    for root, _, files in os.walk('midi'):
        for file in files:
            if file.endswith('.mid') or file.endswith('.midi'):
                try:
                    log_info('Trying to open \"{}\".'.format(file))
                    midi = MidiFile(root + '\\' + file)
                    songs.append(SongFun(msg for _, track in enumerate(midi.tracks) for msg in track))
                    log_ok('Converted correctly.')
                except Exception as e:
                    log_error('Exception during converting \"{}\": {}.'.format(file, str(e)))
    return songs


def get_metadata(midi: MidiFile) -> Tuple[int, str]:
    is_key_signature = False
    is_time_signature = False
    multiple_time_signatures = False

    key_signature = ''
    ticks_per_beat = midi.ticks_per_beat
    ticks_per_measure = 4 * ticks_per_beat

    for i, track in enumerate(midi.tracks):
        for msg in track:
            if msg.type == 'key_signature':
                key_signature = msg.key
                is_key_signature = True
            if msg.type == 'time_signature':
                new_tpm = msg.numerator * ticks_per_beat * 4 / msg.denominator

                if is_time_signature and new_tpm != ticks_per_measure:
                    multiple_time_signatures = True
                else:
                    is_time_signature = True

                ticks_per_measure = new_tpm

    if not is_key_signature:
        raise Exception('No key signature.')
    if not is_time_signature:
        log_warning('No time signature.')
    if multiple_time_signatures:
        log_warning('Multiple time signatures.')

    return ticks_per_measure, key_signature


def midi_to_song(midi: MidiFile) -> Song:
    tpm, key = get_metadata(midi)
    cumulative_time = 0
    first_note = False
    song = Song(key)

    for i, track in enumerate(midi.tracks):
        for msg in track:
            cumulative_time += msg.time
            if msg.type == 'note_on':
                if not first_note:
                    cumulative_time = 0
                    first_note = True
                if msg.velocity != 0:
                    if not song.add_note(cumulative_time, tpm, msg.note):
                        return song
    return song


def song_to_midi(song: Song) -> MidiFile:
    return MidiFile()
