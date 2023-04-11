import torch
from fingerprint.model import FingerprinterModel
from fingerprint.dataset import FingerprintDataset
from tqdm import tqdm
import pickle


def build_db():
    batch_size = 128
    dataset = FingerprintDataset(
        "data/database_recordings", "data/ESC-50-master", "data/IR", batch_size
    )

    # Load Model
    model = FingerprinterModel.load_from_checkpoint(
        "best.ckpt", map_location=torch.device("cpu")
    )
    model.eval()

    db_index = [0] * len(dataset)
    embeds = [0] * len(dataset)
    with torch.no_grad():
        for i, (x_org, x_rep, label) in tqdm(enumerate(dataset), total=len(dataset)):
            z_org, z_rep = model(x_org.unsqueeze(0), x_rep.unsqueeze(0))
            embeds[i] = z_org
            db_index[i] = label

    with open("index2lab.pickle", "wb") as handle:
        pickle.dump(db_index, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("embeds.pickle", "wb") as handle:
        pickle.dump(embeds, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    build_db()
