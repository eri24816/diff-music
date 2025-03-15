from pathlib import Path
import torch
from torch.utils.data import Dataset
import music_data_analysis

class SegmentIndexer:
    def __init__(self, num_segments_list: list[int]):
        self.num_segments_list = torch.tensor(num_segments_list)
        self.length = int(self.num_segments_list.sum().item())
        self.cum_num_segments = torch.cumsum(self.num_segments_list, dim=0)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        song_idx = torch.searchsorted(self.cum_num_segments, idx + 1)
        segment_idx = idx - self.cum_num_segments[song_idx] + self.num_segments_list[song_idx]
        return int(song_idx.item()), int(segment_idx.item())

class PianorollDataset(Dataset):
    def __init__(self, dataset_path: Path, frames_per_beat: int = 8, hop_length: int = 32, length: int = 32*8):
        self.ds = music_data_analysis.Dataset(dataset_path)
        self.frames_per_beat = frames_per_beat
        self.hop_length = hop_length
        self.length = length

        self.songs = self.ds.songs()
        self.song_n_segments = []
        for song in self.songs:
            duration: int = song.read_json('duration') * self.frames_per_beat // 64 # the duration is in 1/64 beat
            self.song_n_segments.append((duration - self.length) // self.hop_length)

        self.indexer = SegmentIndexer(self.song_n_segments)

        print('PianorollDataset initialized with', len(self), 'segments from', len(self.songs), 'songs')

    def __len__(self):
        return len(self.indexer)

    def __getitem__(self, idx):
        song_idx, segment_idx = self.indexer[idx]

        midi = self.songs[song_idx].read_midi('synced_midi')
        pr = music_data_analysis.Pianoroll.from_midi(midi, frames_per_beat=self.frames_per_beat)

        start_time = segment_idx * self.hop_length  
        end_time = start_time + self.length
        result = pr.to_tensor(binary=True, start_time=start_time, end_time=end_time)
        if result.shape[0] != self.length:
            print(f'warning:length mismatch: {result.shape[1]} != {self.length} {self.songs[song_idx].song_name}')
            result = result[:self.length]
            result = torch.cat([result, torch.zeros(self.length - result.shape[0], result.shape[1])], dim=0)
        return result