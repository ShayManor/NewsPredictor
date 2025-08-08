# 1. Feed article summaries through all‑MPNet‑base‑v2 for 768d vectors
# 2. Use datasketch at Jaccard >= 0.8 to remove dupes
# 3. Cluster with HDBSCAN min_cluster_size=5 min_samples=2 and manually label
import os

from cuml.cluster  import HDBSCAN
import numpy as np
import cupy as cp
import torch
from cuml.manifold import UMAP
from bertopic import BERTopic
from datasets import load_dataset
import pandas as pd
from matplotlib import pyplot as plt
from sentence_transformers import SentenceTransformer
torch.set_float32_matmul_precision('high')

os.environ['TOKENIZERS_PARALLELISM'] = "True"

print("Pulling dataset")
ds = load_dataset("LogeshChandran/newsroom", split="train[:100000]").to_pandas()
ds["date"] = pd.to_datetime(ds["date"], format="%Y%m%d", errors="coerce")
print("Sorting dates and building window")
ds = ds.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

# 28 day rolling window
ds["window_id"] = (ds["date"]
                   .dt.floor("28D")
                   .astype("int64") // 10 ** 9)

docs = (ds["title"].fillna("") + " " + ds["summary"].fillna("")).tolist()
timestamps = ds["date"]

assert (len(docs) == len(timestamps))

print("Encoding docs")
encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda", trust_remote_code=True).half()
emb = encoder.encode(docs, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
emb_gpu = cp.asarray(emb)

print("Fitting clusters")
umap_model = UMAP(n_neighbors=20, n_components=15,
                       metric="cosine", min_dist=0.0, random_state=42)

hdbscan_model = HDBSCAN(
    min_cluster_size=8,  # try 6, 8, 10
    min_samples=2,  # 1–5; higher -> more noise, fewer clusters
    metric="euclidean",  # UMAP output is Euclidean
    cluster_selection_epsilon=0.03,  # small epsilon helps attach borderline points
    cluster_selection_method="eom",
    prediction_data=True,
    gen_min_span_tree=True,

)
topic_model = BERTopic(
    embedding_model=None,  # we pass precomputed embeddings
    umap_model=None,
    hdbscan_model=hdbscan_model,
    min_topic_size=8,  # keep aligned with HDBSCAN
    top_n_words=10,
    calculate_probabilities=True,
    verbose=True,
)

topics, probs = topic_model.fit_transform(docs, embeddings=emb)
if probs is not None:
    probs = np.asarray(probs)
    best = probs.max(axis=1)
    best_topic = probs.argmax(axis=1)
    reassign_mask = (np.array(topics) == -1) & (best >= 0.25)  # try 0.20–0.35
    topics = np.where(reassign_mask, best_topic, topics).tolist()
topic_model.reduce_topics(docs, topics=topics, nr_topics=None, similarity_threshold=0.85)

topics_over_time = topic_model.topics_over_time(docs, topics, timestamps, nr_bins=30)

fig = topic_model.visualize_topics_over_time(topics_over_time)
print(topics)
print("=========================================")
print(topics_over_time)


