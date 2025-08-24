# 1. Feed article summaries through all‑MPNet‑base‑v2 for 768d vectors
# 2. Use datasketch at Jaccard >= 0.8 to remove dupes
# 3. Cluster with HDBSCAN and label
# Best run from sweep: {'n_neighbors': 38, 'n_components': 19, 'min_dist': 0.26222776670842407, 'min_c
# luster_size': 4, 'min_samples': 1, 'cluster_epsilon': 0.19291905969822112, 'min_
# topic_size': 16, 'reassign_threshold': 0.11078554780289114}
import json
import os

import numpy as np
from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP
from bertopic import BERTopic
from datasets import load_dataset
import pandas as pd
from matplotlib import pyplot as plt
from sentence_transformers import SentenceTransformer

print("New trial")
os.environ['TOKENIZERS_PARALLELISM'] = "True"

print("Pulling dataset")
ds = load_dataset("LogeshChandran/newsroom", split="train").to_pandas()
print(len(ds))
datetime = pd.to_datetime(ds["date"], format="%Y%m%d", errors="coerce")
mask = datetime.notna()
min_ts = datetime[mask].min()
datetime = datetime.mask(datetime == min_ts, min_ts + pd.Timedelta('1ns'))
print("Sorting dates and building window")
ds = ds.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

# 28 day rolling window
ds["window_id"] = (ds["date"]
                   .dt.floor("28D")
                   .astype("int64") // 10 ** 9)

docs = (ds["title"].fillna("") + " " + ds["summary"].fillna("")).tolist()

assert (len(docs) == len(datetime))

print("Encoding docs")
encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda", trust_remote_code=True).half()
emb = encoder.encode(docs, batch_size=256, show_progress_bar=True, normalize_embeddings=True)

print("Fitting clusters")


def fit_clusters(n_neighbors, n_components, min_dist, min_cluster_size, min_samples, cluster_epsilon, min_topic_size,
                 top_n_words, nr_bins, threshold):
    umap_model = UMAP(n_neighbors=n_neighbors, n_components=n_components,
                      metric="cosine", min_dist=min_dist, random_state=None, build_algo='nn_descent')
    print("Mapped Clusters")

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,  # try 6, 8, 10
        min_samples=min_samples,  # 1–5; higher -> more noise, fewer clusters
        metric="euclidean",  # UMAP output is Euclidean
        cluster_selection_epsilon=cluster_epsilon,  # small epsilon helps attach borderline points
        cluster_selection_method="eom",
        prediction_data=False,
        gen_min_span_tree=False,
    )
    print("Scanned clusters")

    topic_model = BERTopic(
        embedding_model=None,  # we pass precomputed embeddings
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        min_topic_size=min_topic_size,  # keep aligned with HDBSCAN
        top_n_words=top_n_words,
        calculate_probabilities=False,
        verbose=False,
        low_memory=True,
    )

    topics, probs = topic_model.fit_transform(docs, embeddings=emb)
    print("Fit embeddings")
    topics = topic_model.reduce_outliers(docs, topics, strategy="embeddings", embeddings=emb, threshold=threshold)
    print("Updated topics")
    topic_model.update_topics(docs, topics=topics)
    print("Reduced outliers")
    topics = [int(x) for x in topics]
    jsonpath = 'topics_array.json'
    if not os.path.exists(jsonpath):
        open(jsonpath, 'x')
    with open(jsonpath, 'w') as f:
        json.dump(topics, f)
    print(f"Topics: {topics}")
    res = {}
    for idx, topic in enumerate(topics):
        res[docs[idx]] = topic
    results_path = "results.json"
    if not os.path.exists(results_path):
        open(results_path, 'x')
    with open(results_path) as f:
        json.dump(res, f)
    return topics.count(-1)


print(
    f"Result: {fit_clusters(52, 23, 0.010292247147901223, 20, 5, 0.04246782213565523, 7, 10, threshold=0.13253202943404524, nr_bins=30)}")

# if __name__ == '__main__':
# {'n_neighbors': 38, 'n_components': 19, 'min_dist': 0.26222776670842407, 'min_c
# luster_size': 4, 'min_samples': 1, 'cluster_epsilon': 0.19291905969822112, 'min_
# topic_size': 16, 'reassign_threshold': 0.11078554780289114}

# {'n_neighbors': 52, 'n_components': 23, 'min_dist': 0.010292247147901223, 'min_cluster_size': 20, 'min_samples': 5,
#  'cluster_epsilon': 0.04246782213565523, 'min_topic_size': 7, 'reassign_threshold': 0.13253202943404524}
