import torch
from fingerprint.model import FingerprinterModel, similarity_loss
from fingerprint.dataset import FingerprintDataset
import pickle
from tqdm import tqdm


# Fix Chunks_per_song
def main():
    dataset = FingerprintDataset(
        "data/query_recordings", "data/ESC-50-master", "data/IR", length=10
    )
    # dataset = FingerprintDataset(
    #     "data/database_recordings", "data/ESC-50-master", "data/IR", length=30
    # )
    model = FingerprinterModel.load_from_checkpoint(
        "best.ckpt", map_location=torch.device("cpu")
    )
    model.eval()

    with open("index2lab.pickle", "rb") as handle:
        db_index = pickle.load(handle)
    with open("embeds.pickle", "rb") as handle:
        embeds = pickle.load(handle)

    with torch.no_grad():
        scores = []

        chunk_n = 0
        num_correct = 0
        num_top1_correct = 0
        for i, (x_org, x_rep, label) in tqdm(enumerate(dataset), total=len(dataset)):
            z_org, z_rep = model(x_org.unsqueeze(0), x_rep.unsqueeze(0))
            search_results = get_top_k(z_org, embeds, db_index, k=3)
            scores.extend(search_results)
            chunk_n += 1
            if chunk_n == dataset.chunks_per_song:
                scores.sort(key=lambda x: x[0], reverse=True)
                score_names = set()
                final_scores = []
                for score, db_label in scores:
                    if db_label not in score_names:
                        score_names.add(db_label)
                        final_scores.append(score)
                    if len(score_names) == 3:
                        break
                print(f"Query: {label}")
                correct = False
                for i, (score, db_label) in enumerate(zip(final_scores, score_names)):
                    if i == 0:
                        if label.split("-")[0] == db_label:
                            num_top1_correct += 1
                    print(f"Score: {score} Label: {db_label}")
                    if label.split("-")[0] == db_label:
                        # if label == db_label:
                        correct = True
                        num_correct += 1
                print(f"Correct: {correct}")
                chunk_n = 0
                scores = []
        print(
            f"Accuracy top 1: {round(num_top1_correct / (len(dataset) / dataset.chunks_per_song), 2)}"
        )
        print(
            f"Accuracy top 3: {round(num_correct / (len(dataset) / dataset.chunks_per_song), 2)}"
        )


def get_top_k(query, embeds, db_index, k=3):
    scores = []
    for i, embed in enumerate(embeds):
        score = torch.cosine_similarity(query, embed, dim=1)
        scores.append((score, db_index[i]))
    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[:k]


if __name__ == "__main__":
    main()
