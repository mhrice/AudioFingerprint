import pytorch_lightning as pl
from fingerprint.dataset import FingerprintDataModule, FingerprintDataset
from fingerprint.model import FingerprinterModel
from pytorch_lightning.loggers import WandbLogger
import torch


def main():
    batch_size = 128
    dataset = FingerprintDataset(
        "data/database_recordings", "data/ESC-50-master", "data/IR", batch_size
    )
    torch.set_float32_matmul_precision("medium")
    data = FingerprintDataModule(dataset, batch_size, num_workers=2)
    model = FingerprinterModel()
    wandb_logger = WandbLogger(project="Fingerprint", entity="mattricesound")
    trainer = pl.Trainer(max_epochs=100, accelerator="gpu", logger=wandb_logger)
    trainer.fit(model, data)


if __name__ == "__main__":
    main()
