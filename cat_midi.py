import os
from pathlib import Path

from music_data_analysis import Pianoroll

input_dir = 'wandb/run-20250317_145030-20250317_145030/files'

output = './all.mid'


midi_files = [f for f in os.listdir(input_dir) if f.endswith('.mid')]
midi_files = [f for f in midi_files if int(f.split('_')[-1].split('.')[0]) > 180000]

# Sort the midi files to ensure consistent ordering
midi_files.sort()


all_pr = []
for midi_file in midi_files:
    print('Adding', midi_file)
    pr = Pianoroll.from_midi(Path(input_dir) / midi_file)
    all_pr.append(pr)

result = all_pr[0]
for pr in all_pr[1:]:
    result |= pr

result.to_midi(output)
