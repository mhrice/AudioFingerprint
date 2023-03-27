import pytorch_lightning as pl
from fingerprint.dataset import FingerprintDataModule, FingerprintDataset
from fingerprint.model import FingerprinterModel


def main():
    batch_size = 32
    dataset = FingerprintDataset(
        "data/database_recordings", "data/ESC-50-master", "data/IR", batch_size
    )
    data = FingerprintDataModule(dataset, batch_size)
    model = FingerprinterModel()

    trainer = pl.Trainer(max_epochs=100, accelerator="gpu")
    trainer.fit(model, data)


if __name__ == "__main__":
    main()
