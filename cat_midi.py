import os

input_dir = 'wandb/run-20250317_003641-20250317_003641/files'

output = './all.mid'

import pretty_midi

midi_files = [f for f in os.listdir(input_dir) if f.endswith('.mid')]
midi_files = [f for f in midi_files if int(f.split('_')[-1].split('.')[0]) > 41200]

# Sort the midi files to ensure consistent ordering
midi_files.sort()

# Create a new MIDI object to hold all the concatenated data
combined_midi = pretty_midi.PrettyMIDI()

# Track the total duration to offset each subsequent MIDI file
current_offset = 0.0

# Process each MIDI file
for midi_file in midi_files:
    # Load the MIDI file
    midi_path = os.path.join(input_dir, midi_file)
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
        
        # Get the end time of this MIDI file
        if len(midi.instruments) > 0:
            file_duration = max([note.end for instrument in midi.instruments for note in instrument.notes]) if any(instrument.notes for instrument in midi.instruments) else 0
        else:
            file_duration = 0
        
        # Add each instrument from this MIDI file to the combined MIDI
        for instrument in midi.instruments:
            # Create a new instrument of the same type
            new_instrument = pretty_midi.Instrument(program=instrument.program, is_drum=instrument.is_drum)
            
            # Add all notes with offset
            for note in instrument.notes:
                new_note = pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=note.start + current_offset,
                    end=note.end + current_offset
                )
                new_instrument.notes.append(new_note)
            
            # Add all control changes with offset
            for cc in instrument.control_changes:
                new_cc = pretty_midi.ControlChange(
                    number=cc.number,
                    value=cc.value,
                    time=cc.time + current_offset
                )
                new_instrument.control_changes.append(new_cc)
            
            # Add the instrument to the combined MIDI
            combined_midi.instruments.append(new_instrument)
        
        # Update the offset for the next file
        current_offset += file_duration + 1.0  # Add 1 second gap between files
        
        print(f"Added {midi_file} (duration: {file_duration:.2f}s)")
    except Exception as e:
        print(f"Error processing {midi_file}: {e}")

# Write the combined MIDI to the output file
combined_midi.write(output)
print(f"Successfully concatenated {len(midi_files)} MIDI files to {output}")
