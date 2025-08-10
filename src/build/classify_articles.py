# 1. Feed article summaries through all‑MPNet‑base‑v2 for 768d vectors
# 2. Use datasketch at Jaccard >= 0.8 to remove dupes
# 3. Cluster with HDBSCAN and label
# Best run from sweep: {'n_neighbors': 38, 'n_components': 19, 'min_dist': 0.26222776670842407, 'min_c
# luster_size': 4, 'min_samples': 1, 'cluster_epsilon': 0.19291905969822112, 'min_
# topic_size': 16, 'reassign_threshold': 0.11078554780289114}
import json
import os

from cuml.cluster import HDBSCAN
import numpy as np
from cuml.manifold import UMAP
from bertopic import BERTopic
from datasets import load_dataset
import pandas as pd
from matplotlib import pyplot as plt
from sentence_transformers import SentenceTransformer
import optuna

print("New trial")
os.environ['TOKENIZERS_PARALLELISM'] = "True"

print("Pulling dataset")
ds = load_dataset("LogeshChandran/newsroom", split="train").to_pandas()
ds["date"] = pd.to_datetime(ds["date"], format="%Y%m%d", errors="coerce")
print("Sorting dates and building window")
ds = ds.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

# 28 day rolling window
ds["window_id"] = (ds["date"]
                   .dt.floor("28D")
                   .astype("int64") // 10 ** 9)

docs = (ds["title"].fillna("") + " " + ds["summary"].fillna("")).tolist()
timestamps = ds["date"].dt.to_pydatetime().tolist()

assert (len(docs) == len(timestamps))

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
    )

    topics, probs = topic_model.fit_transform(docs, embeddings=emb)
    print("Fit embeddings")
    topics = topic_model.reduce_outliers(docs, topics, probabilities=probs, strategy="c-tf-idf", threshold=threshold)
    topic_model.update_topics(docs, topics=topics)

    topic_model.merge_topics(docs, topics)
    topic_model.update_topics(docs, topics=topics)

    topics_over_time = topic_model.topics_over_time(docs=docs, topics=topics, timestamps=timestamps, nr_bins=nr_bins,
                                                    global_tuning=False)

    fig = topic_model.visualize_topics_over_time(topics_over_time)
    fig.show()
    with open('topics_array.json', 'w') as f:
        json.dump(topics, f)
    print(f"Topics: {topics}")
    print("=========================================")
    print(f"Topics over time: {topics_over_time}")
    return topics.count(-1)


fit_clusters(38, 19, 0.26222776670842407, 4, 1, 0.19291905969822112, 16, 10, threshold=0.11078554780289114, nr_bins=30)

# if __name__ == '__main__':
# {'n_neighbors': 38, 'n_components': 19, 'min_dist': 0.26222776670842407, 'min_c
# luster_size': 4, 'min_samples': 1, 'cluster_epsilon': 0.19291905969822112, 'min_
# topic_size': 16, 'reassign_threshold': 0.11078554780289114}
