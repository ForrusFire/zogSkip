import glob
import pickle
import music21 as m21


PATH_TO_CLASSICAL_MIDI = "data/raw_midi/classical_midi"
PATH_TO_SAVE_NOTES = "data/preprocessed/notes"


def preprocess(dataset_path, save_path):
    # Load songs from dataset path
    print("Loading songs...")
    songs = load_songs(dataset_path)
    print(f"Loaded {len(songs)} songs")

    # Parse loaded songs
    print("Parsing...")
    notes = parse_songs(songs)
    print("Finished parsing")

    # Save parsed notes
    print("Saving parsed notes")
    save_notes(notes, save_path)
    
    
def load_songs(dataset_path):
    songs = []

    # Convert all midi files from the dataset path into m21 Stream objects
    for file in glob.glob(dataset_path + "/*.mid"):
        song = m21.converter.parse(file)
        songs.append(song)

    return songs


def parse_songs(songs):
    notes = []

    for song in songs:
        notes_to_parse = None

        try:
            # If file has instrument parts, extract the first instrument's notes (piano)
            instruments = m21.instrument.partitionByInstrument(song)
            notes_to_parse = instruments.parts[0].recurse()
        except:
            # Otherwise, collect the song's notes
            notes_to_parse = song.flat.notesAndRests

        # Parse each element
        for element in notes_to_parse:
            if isinstance(element, m21.note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, m21.chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
            elif isinstance(element, m21.note.Rest):
                notes.append(element.name)

    return notes


def save_notes(notes, save_path):
    # Save notes into save path
    with open(save_path, 'wb') as fp:
        pickle.dump(notes, fp)



if __name__ == "__main__":
    preprocess(PATH_TO_CLASSICAL_MIDI, PATH_TO_SAVE_NOTES)