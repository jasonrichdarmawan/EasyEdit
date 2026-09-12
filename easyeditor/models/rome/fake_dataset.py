import torch
from torch.utils.data import Dataset


class FakeTokenDataset(Dataset):
    """Fixed-length token sequences for testing model VRAM usage."""
    def __init__(self, sample_count, sequence_length):
        self.sample_count = sample_count
        self.sequence_length = sequence_length

    def __len__(self):
        return self.sample_count

    def __getitem__(self, idx):
        return {
            "input_ids": torch.ones((self.sequence_length,), dtype=torch.long),
            "attention_mask": torch.ones((self.sequence_length,), dtype=torch.long),
        }