import pytorch_lightning as pl
from fingerprint.dataset import FingerprintDataModule, FingerprintDataset
from fingerprint.model import FingerprinterModel
from pytorch_lightning.loggers import WandbLogger
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

length = 30
batch_size = 128


def main():
    dataset = FingerprintDataset(
        "data/database_recordings", "data/ESC-50-master", "data/IR", length
    )
    torch.set_float32_matmul_precision("medium")
    data = FingerprintDataModule(dataset, batch_size, num_workers=2)
    model = FingerprinterModel()
    checkpoint_callback = ModelCheckpoint(
        dirpath="./ckpts", save_top_k=1, monitor="val_loss"
    )
    wandb_logger = WandbLogger(project="Fingerprint", entity="mattricesound")
    trainer = pl.Trainer(
        max_epochs=200,
        accelerator="gpu",
        logger=wandb_logger,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, data)


if __name__ == "__main__":
    main()
