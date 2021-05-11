import pickle

import torch
from torch.utils.data import Dataset


class NotesDataset(Dataset):
    def __init__(self, notes_path, sequence_length):
        self.input, self.output = prepare_sequences(notes_path, sequence_length)

    def __getitem__(self, index):
        return (self.input[index], self.output[index])

    def __len__(self):
        return len(self.input)


def prepare_sequences(notes_path, sequence_length):
    notes = load_notes(notes_path)

    # Create a dictionary that maps notes to ints
    note_to_int = create_note_to_int_dict(notes)

    # Dataset input and output lists
    network_input = []
    network_output = []

    # Fill input and output lists
    for i in range(len(notes) - sequence_length):
        input_sequence = notes[i: i+sequence_length]
        output_sequence = notes[i+sequence_length]

        network_input.append([note_to_int[note] for note in input_sequence])
        network_output.append(note_to_int[output_sequence])

    # Convert lists into PyTorch tensors
    network_input = torch.FloatTensor(network_input)
    network_output = torch.FloatTensor(network_output)

    return (network_input, network_output)


def load_notes(notes_path):
    # Load notes from notes path
    with open(notes_path, 'rb') as fp:
        notes = pickle.load(fp)

    return notes


def create_note_to_int_dict(notes):
    # Create sorted set of distinct note pitches and rests
    note_names = sorted(set(note for note in notes))

    # Create the dict
    note_to_int = dict((note, number) for number, note in enumerate(note_names))

    return note_to_int
