import pytorch_lightning as pl
from fingerprint.dataset import FingerprintDataModule, FingerprintDataset
from fingerprint.model import FingerprinterModel


def main():
    model = FingerprinterModel()
    dataset = FingerprintDataset("sss_free")
    batch_size = 32
    data = FingerprintDataModule(dataset, batch_size)

    trainer = pl.Trainer(max_epochs=10, accelerator="cpu")
    trainer.fit(model, data)


if __name__ == "__main__":
    main()
