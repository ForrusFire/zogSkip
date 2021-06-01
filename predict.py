import numpy as np
import torch

import music21 as m21

import train as tn
from dataset import prepare_sequences

Model = tn.Model
epochs = 100


PATH_TO_NOTES = tn.PATH_TO_NOTES
PATH_TO_WEIGHTS = tn.PATH_TO_SAVE_WEIGHTS + "_" + str(epochs) + "epochs.pth"
PATH_TO_SAVE_MIDI = "predictions/" + tn.MODEL_NAME + "_" + tn.DATASET + "_" + tn.PARAMETER_SET + "/predict"
NUM_MIDI_FILES = 10
NUM_GENERATED_NOTES = 500


def predict(notes_path, weights_path, save_path):
    """
    Generate a piano midi file using the trained model
    """
    print("Predicting...")
    int_to_note, sequences_list = prepare_sequences(notes_path, tn.sequence_length, predict=True)
    
    # Loads the model
    num_classes = len(int_to_note)
    model = load_model(weights_path, num_classes)

    for i in range(NUM_MIDI_FILES):
        # Creates the prediction and turns it into a midi file
        print("Generating notes...")
        notes = generate_notes(model, int_to_note, sequences_list)
        print("Creating midi...")
        midi = create_midi(notes)

        print("Saving midi...")
        full_save_path = save_path + f"_{i+1}.mid"
        save_midi(midi, full_save_path)
        print(f"{i+1} of {NUM_MIDI_FILES} midi files generated...")

    print("Prediction complete!")


def load_model(weights_path, num_classes):
    # Loads the model
    model = Model(
        tn.LSTM1_num_units,
        tn.SeqSelfAttention_num_units,
        tn.LSTM2_num_units,
        tn.dropout_prob,
        tn.sequence_length,
        num_classes
    ).to(tn.DEVICE)

    model.load_state_dict(torch.load(weights_path))
    model.eval() # Sets model in evaluation mode

    return model


def generate_notes(model, int_to_note, sequences_list):
    # Generates notes from the model, randomly selecting one of the network input sequences to start
    sequence = random_sequence(sequences_list)
    notes = []

    with torch.no_grad():
        for _ in range(NUM_GENERATED_NOTES):
            # Prepare the sequence as a tensor
            sequence_as_tensor = torch.unsqueeze(torch.unsqueeze(torch.FloatTensor(sequence), 0), 2)
            sequence_as_tensor.to(tn.DEVICE)

            # Generate model output and convert to a note
            output = model(sequence_as_tensor)
            note_index = int(torch.argmax(output).item())
            note = int_to_note[note_index]
            notes.append(note)

            # Add the new (normalized) note to the end of the sequence and remove the first note in the sequence
            sequence.append(note_index / len(int_to_note))
            sequence = sequence[1:]

    return notes


def random_sequence(sequences_list):
    # Returns a random sequence from the sequences list
    rand_index = np.random.randint(len(sequences_list))

    return sequences_list[rand_index]


def create_midi(raw_patterns):
    """
    Creates the midi file from the list of note and chord patterns
    """
    offset = 0
    song = []

    # Create note and chord objects based on the list of raw patterns
    for raw_pattern in raw_patterns:
        pattern, duration = raw_pattern.split()

        # If pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            new_chord = create_new_chord(pattern, offset)
            song.append(new_chord)
        # If pattern is a rest
        elif ('rest' in pattern):
            new_rest = create_new_rest(pattern, offset)
            song.append(new_rest)
        # If pattern is a single note
        else:
            new_note = create_new_note(pattern, offset)
            song.append(new_note)

        # Increase offset by the duration of the pattern
        offset += convert_frac_to_float(duration)

    midi = m21.stream.Stream(song)

    return midi


def create_new_chord(raw_chord, offset):
    # Creates a music21 chord object from the raw chord
    notes_in_chord = raw_chord.split('.')

    chord = []
    for note in notes_in_chord:
        new_note = m21.note.Note(int(note))
        new_note.storedInstrument = m21.instrument.Piano()
        chord.append(new_note)

    new_chord = m21.chord.Chord(chord)
    new_chord.offset = offset

    return new_chord


def create_new_rest(raw_rest, offset):
    # Creates a music21 rest note object from the raw rest
    new_rest = m21.note.Rest(raw_rest)
    new_rest.storedInstrument = m21.instrument.Piano()
    new_rest.offset = offset

    return new_rest


def create_new_note(raw_note, offset):
    # Creates a music21 note object from the raw note
    new_note = m21.note.Note(raw_note)
    new_note.storedInstrument = m21.instrument.Piano()
    new_note.offset = offset

    return new_note


def convert_frac_to_float(frac_str):
    # Converts a fraction to a float
    try:
        return float(frac_str)
    except ValueError:
        num, denom = frac_str.split('/')
        try:
            # Try parsing for mixed fraction
            leading, num = num.split(' ')
            whole = float(leading)
        except ValueError:
            whole = 0
        frac = float(num) / float(denom)

        return whole - frac if whole < 0 else whole + frac


def save_midi(midi, save_path):
    # Saves the midi file
    midi.write('midi', fp=save_path)



if __name__ == "__main__":
    predict(PATH_TO_NOTES, PATH_TO_WEIGHTS, PATH_TO_SAVE_MIDI)
