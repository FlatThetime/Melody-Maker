# Import the necessary libraries
import tensorflow as tf
import numpy as np
from music21 import *

# Define the notes of the melody
notes = ['C', 'D', 'E', 'F', 'G', 'A', 'B']

# Define the instruments to use
instruments = ['piano', 'violin', 'trumpet', 'flute']

# Load the trained RNN model
model = tf.keras.models.load_model('classical_music_rnn.h5')

def generate_music(length, temperature, output_file, instrument='piano', tempo=120):
    # Set up the melody
    melody = stream.Stream()

    # Set the starting note of the melody
    start_note = note.Note('C')

    # Loop through the notes and generate the melody
    for i in range(length):
        # Convert the current note to a one-hot encoded vector
        current_note = notes.index(str(start_note.pitch))
        current_note_vector = np.zeros((len(notes),))
        current_note_vector[current_note] = 1

        # Predict the next note using the RNN model
        next_note_vector = model.predict(np.array([current_note_vector]))[0]

        # Apply temperature to the predicted probability distribution
        next_note_vector = np.log(next_note_vector) / temperature
        next_note_vector = np.exp(next_note_vector) / np.sum(np.exp(next_note_vector))

        # Sample the next note from the predicted probability distribution
        next_note = np.random.choice(range(len(notes)), p=next_note_vector)
        next_note = note.Note(notes[next_note])

        # Add the next note to the melody
        melody.append(next_note)

        # Set the starting note for the next iteration
        start_note = next_note

    # Add a time signature, key signature, tempo, and instrument to the melody
    melody.insert(0, meter.TimeSignature('4/4'))
    melody.insert(0, key.Key('C'))
    melody.insert(0, tempo.MetronomeMark(number=tempo))
    melody.insert(0, instrument)

    # Save the melody as a MIDI file
    mf = midi.translate.streamToMidiFile(melody)
    mf.open(output_file, 'wb')
    mf.write()
    mf.close()
    
    print(f"Music generated and saved to {output_file}.")

# Example usage
length = 100
temperature = 0.5
output_file = 'output.mid'
instrument = instrument.Violin
tempo = 80
generate_music(length, temperature, output_file, instrument, tempo)
