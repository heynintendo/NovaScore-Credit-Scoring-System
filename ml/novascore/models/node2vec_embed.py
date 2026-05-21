"""Node2Vec embeddings on the user-merchant bipartite graph.

Each transaction adds an edge (user_id, merchant_id). Node2Vec generates random
walks and learns a 64-d embedding per node via SGNS (skip-gram with negative
sampling). The user-side embeddings are returned in the order of `user_ids`;
users absent from the graph (e.g., unseen at inference) get zero vectors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:  # pragma: no cover - import guard
    import networkx as nx
    from node2vec import Node2Vec
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "node2vec_embed requires networkx + node2vec; pip install networkx node2vec gensim"
    ) from e


def build_user_merchant_graph(txns_w: pd.DataFrame) -> "nx.Graph":
    """Build an undirected bipartite graph from (user_id, merchant_id) edges."""
    edges = txns_w[["user_id", "merchant_id"]].dropna()
    pairs = [(str(u), f"m_{m}") for u, m in edges.itertuples(index=False, name=None)]
    G = nx.Graph()
    G.add_edges_from(pairs)
    return G


def compute_user_embeddings(
    txns_w: pd.DataFrame,
    user_ids: pd.Series,
    dim: int = 64,
    walk_length: int = 20,
    num_walks: int = 100,
    seed: int = 42,
    workers: int = 2,
) -> np.ndarray:
    """Train Node2Vec on the user-merchant graph; return (n_users, dim) array."""
    G = build_user_merchant_graph(txns_w)
    if G.number_of_edges() == 0:
        return np.zeros((len(user_ids), dim), dtype="float32")
    n2v = Node2Vec(
        G,
        dimensions=dim,
        walk_length=walk_length,
        num_walks=num_walks,
        workers=workers,
        seed=seed,
        quiet=True,
    )
    model = n2v.fit(window=10, min_count=1, batch_words=4)
    out = np.zeros((len(user_ids), dim), dtype="float32")
    for i, u in enumerate(user_ids.astype(str).tolist()):
        if u in model.wv:
            out[i] = model.wv[u].astype("float32")
    return out
