import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
from pathlib import Path
import torchaudio
from torch.utils.data import random_split
import torch.nn.functional as F

sample_rate = 8000
chunk_size = int(1.2 * sample_rate)
clipped_chunk_size = int(1.0 * sample_rate)
length = 30
chunk_overlap = 0.5
chunks_per_song = int(length // chunk_overlap)


class FingerprintDataset(Dataset):
    def __init__(self, data_dir, noise_dir, ir_dir, batch_size, anchors_per_batch=8):
        super().__init__()
        self.root = Path(data_dir)
        self.files = list(self.root.glob("*.wav"))
        self.noise_dir = Path(noise_dir) / "audio"
        self.noise_files = list(self.noise_dir.glob("*.wav"))
        self.ir_dir = Path(ir_dir)
        self.ir_files = list(self.ir_dir.glob("*.wav"))
        shuffle_indices = torch.randperm(len(self.files))
        self.files = [self.files[i] for i in shuffle_indices]
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=1024, hop_length=256, n_mels=256
        )

    def __len__(self):
        return len(self.files) * chunks_per_song

    def __getitem__(self, idx):
        song_id = int(idx // (chunks_per_song))
        chunk_id = int(idx % (chunks_per_song))
        data, sr = torchaudio.load(self.files[song_id])
        data = torchaudio.functional.resample(data, sr, sample_rate)
        start = int(chunk_id * chunk_overlap * sample_rate)
        chunk = data[:, start : start + chunk_size]
        # Sum to mono
        if chunk.shape[0] > 1:
            chunk = torch.sum(chunk, dim=0, keepdim=True)
        if chunk.shape[1] < chunk_size:
            chunk = F.pad(chunk, (0, chunk_size - chunk.shape[1]))

        # pick random 1s portion
        start1 = torch.randint(0, chunk.shape[1] - clipped_chunk_size, (1,))
        start2 = torch.randint(0, chunk.shape[1] - clipped_chunk_size, (1,))
        x_org = chunk[:, start1 : start1 + clipped_chunk_size]
        x_rep = chunk[:, start2 : start2 + clipped_chunk_size]

        # TD Augmentations
        # Background mixing
        noise_file = self.noise_files[torch.randint(0, len(self.noise_files), (1,))]
        noise, sr = torchaudio.load(noise_file)
        noise = torchaudio.functional.resample(noise, sr, sample_rate)
        if noise.shape[0] > 1:
            noise = torch.sum(noise, dim=0, keepdim=True)
        if noise.shape[1] < chunk_size:
            noise = F.pad(noise, (0, chunk_size - noise.shape[1]))
        random_noise_start = torch.randint(0, noise.shape[1] - clipped_chunk_size, (1,))
        noise = noise[:, random_noise_start : random_noise_start + clipped_chunk_size]
        x_rep = x_rep + noise
        # IR Filter
        ir_file = self.ir_files[torch.randint(0, len(self.ir_files), (1,))]
        ir, sr = torchaudio.load(ir_file)
        ir = torchaudio.functional.resample(ir, sr, sample_rate)
        if ir.shape[0] > 1:
            ir = torch.sum(ir, dim=0, keepdim=True)
        # Convolve
        fftLength = x_rep.shape[1]
        X = torch.fft.fft(x_rep, n=fftLength)
        X_ir = torch.fft.fft(ir, n=fftLength)
        x_rep = torch.fft.ifft(X_ir * X)[:fftLength].real
        # Spectrogram
        X_org = self.mel(x_org)  # C x F x T
        X_rep = self.mel(x_rep)  # C x F x T
        # Spectrogram Masking
        msk = torch.zeros(
            (int(round(X_rep.shape[1] / 2)), int(round(X_rep.shape[2] / 10)))
        )
        msk_x = torch.randint(0, X_rep.shape[1] - msk.shape[0], (1,))
        msk_y = torch.randint(0, X_rep.shape[2] - msk.shape[1], (1,))
        X_rep[:, msk_x : msk_x + msk.shape[0], msk_y : msk_y + msk.shape[1]] = msk
        return X_org, X_rep


class FingerprintDataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

    def setup(self, stage=None):
        train_split = int(len(self.dataset) * 0.75)
        val_split = len(self.dataset) - train_split
        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [train_split, val_split]
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False
        )
