from typing import Tuple

from mido import MidiFile, MidiTrack, Message

from song import Song, ATOMS_IN_MEASURE, MIN_NOTE
from logger import *


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


def song_to_midi(song: Song, filename: str):
    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)
    track.append(Message('program_change', program=1, time=0))

    ticks_per_beat = midi.ticks_per_beat
    ticks_per_measure = 4 * ticks_per_beat
    ticks_per_atom = int(ticks_per_measure / ATOMS_IN_MEASURE)

    last_time = 0

    notes_on = [(measure * ATOMS_IN_MEASURE + atom, pitch, 'note_on') for [measure, atom, pitch] in song.data.nonzero()]
    notes_off = [(atom + 1, pitch, 'note_off') for (atom, pitch, _) in notes_on]
    notes = sorted(notes_on + notes_off)

    for (atom, pitch, message) in notes:
        atom, pitch = int(atom), int(pitch)
        abs_time = atom * ticks_per_atom
        delta_time = abs_time - last_time
        note = pitch + MIN_NOTE
        track.append(Message(message, note=note, velocity=100, time=delta_time))
        last_time = abs_time

    midi.save('{}.mid'.format(filename))
