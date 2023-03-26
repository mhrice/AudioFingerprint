import pytorch_lightning as pl
from fingerprint.dataset import FingerprintDataModule, FingerprintDataset
from fingerprint.model import FingerprinterModel


def main():
    dataset = FingerprintDataset("data/database_recordings", 16)
    batch_size = 32
    data = FingerprintDataModule(dataset, batch_size)
    model = FingerprinterModel()

    trainer = pl.Trainer(max_epochs=10, accelerator="cpu")
    trainer.fit(model, data)


if __name__ == "__main__":
    main()
