# Builds corpus of news articles on the same story in sequential order
import faiss
from line_profiler_pycharm import profile
from datasets import load_dataset
import logging

from datasketch import MinHashLSH, MinHash
import numpy as np
import umap
import hdbscan
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


@profile
def pull_dataset():
    logger.info("Pulling Dataset")
    ds = load_dataset("LogeshChandran/newsroom")["train"][:10000]
    logger.info("Embedding summaries")
    summaries = list(ds["summary"])
    model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-v4').bfloat16()
    return model, summaries


@profile
def encode(model, summaries):
    logger.info("Removing dupes")
    embeddings = model.encode(summaries)
    emb = np.array(embeddings).astype('float32')
    emb = np.ascontiguousarray(emb)
    index = faiss.IndexFlatIP(emb.shape[1])
    faiss.normalize_L2(emb)
    index.add(emb)
    D, I = index.search(emb, k=5)
    dupes = set()
    first = True
    for i, (d, id_) in enumerate(zip(D, I)):
        if d[1] > 0.85:
            dupes.add(i)
            if first:
                for idx in id_:
                    print(f"{idx}: {summaries[idx]}")
                    first = False
            print(f"Dupe: {i}, {id_}")
    print(len(dupes))
    return emb, dupes


def cluster(embs, dupes):
    mask = np.ones(len(embs), dtype=bool)
    mask[list(dupes)] = False
    emb_nd = embs[mask]

    clusterer = hdbscan.HDBSCAN(min_cluster_size=8, cluster_selection_method='leaf', metric='euclidean', allow_single_cluster=True)
    clusterer.fit(emb_nd)

    labels = clusterer.labels_
    print(f"{labels.max() + 1} clusters, {np.sum(labels == -1)} noise points")

    print(labels)
    proj = umap.UMAP(n_neighbors=15, min_dist=0.1).fit_transform(emb_nd)
    plt.scatter(proj[:, 0], proj[:, 1], c=labels, cmap="viridis", s=8)
    plt.title("HDBSCAN clusters on MPNet summaries")
    plt.savefig('clusters.png')
    plt.show()


if __name__ == '__main__':
    model, summaries = pull_dataset()
    embs, dupes = encode(model, summaries)
    cluster(embs, dupes)
