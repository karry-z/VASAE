"""Cluster and visualize GPT-2 embedding vectors (token directions).

Usage:
    python scripts/cluster_embeddings.py \
        --blackbox-model-dir /scratch/b5bq/pu22650.b5bq/VASAE_out/BlackBoxModels/gpt2 \
        --n-clusters 20 \
        --method kmeans \
        --dim-reduction tsne \
        --output-dir outputs/emb_clusters
"""

import os

# Must be set before numpy / OpenBLAS is imported
os.environ.setdefault("OPENBLAS_NUM_THREADS", "64")
os.environ.setdefault("OMP_NUM_THREADS", "64")

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from vasae.models.factory import BlackBoxModelConfig, load_embedding_layer

# ---------- helpers ----------------------------------------------------------


def get_embeddings(cfg: BlackBoxModelConfig) -> np.ndarray:
    """Load embedding layer and return weight matrix as numpy (vocab x dim)."""
    emb = load_embedding_layer(cfg)
    return emb.weight.detach().cpu().numpy()


def cluster_embeddings(
    vecs: np.ndarray,
    method: str = "kmeans",
    n_clusters: int = 20,
    eps: float = 0.5,
    min_samples: int = 5,
) -> np.ndarray:
    """Return cluster labels for each embedding vector."""
    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    elif method == "agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters)
    elif method == "dbscan":
        model = DBSCAN(eps=eps, min_samples=min_samples)
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    labels = model.fit_predict(vecs)
    return labels


def reduce_dims(
    vecs: np.ndarray,
    method: str = "tsne",
    n_components: int = 2,
    pca_before_tsne: int = 50,
) -> np.ndarray:
    """Reduce high-dimensional vectors to 2-D (or 3-D) for visualization."""
    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=42)
        return reducer.fit_transform(vecs)

    if method == "tsne":
        # PCA first to speed up t-SNE
        if vecs.shape[1] > pca_before_tsne:
            vecs = PCA(n_components=pca_before_tsne, random_state=42).fit_transform(
                vecs
            )
        reducer = TSNE(
            n_components=n_components,
            random_state=42,
            perplexity=min(30, len(vecs) - 1),
            init="pca",
            learning_rate="auto",
        )
        return reducer.fit_transform(vecs)

    raise ValueError(f"Unknown dim-reduction method: {method}")


# ---------- tokenizer (optional, for annotation) ----------------------------

_tokenizer = None


def get_tokenizer(model_name: str = "gpt2"):
    global _tokenizer
    if _tokenizer is None:
        from transformers import GPT2TokenizerFast

        _tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    return _tokenizer


# ---------- plotting ---------------------------------------------------------


def plot_clusters_2d(
    coords_2d: np.ndarray,
    labels: np.ndarray,
    *,
    title: str = "GPT-2 Embedding Clusters",
    save_path: Path | None = None,
    annotate_n: int = 0,
    token_strs: list[str] | None = None,
):
    """Scatter plot of 2-D projections coloured by cluster label."""
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    cmap = plt.cm.get_cmap("tab20", max(n_clusters, 20))

    fig, ax = plt.subplots(figsize=(14, 10))
    for idx, lbl in enumerate(unique_labels):
        mask = labels == lbl
        colour = "grey" if lbl == -1 else cmap(idx % 20)
        label_str = "noise" if lbl == -1 else f"cluster {lbl}"
        ax.scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            s=4,
            alpha=0.6,
            c=[colour],
            label=label_str,
        )

    # optionally annotate a few tokens per cluster
    if annotate_n > 0 and token_strs is not None:
        rng = np.random.default_rng(42)
        for lbl in unique_labels:
            if lbl == -1:
                continue
            idxs = np.where(labels == lbl)[0]
            chosen = rng.choice(idxs, size=min(annotate_n, len(idxs)), replace=False)
            for i in chosen:
                ax.annotate(
                    token_strs[i],
                    (coords_2d[i, 0], coords_2d[i, 1]),
                    fontsize=6,
                    alpha=0.8,
                )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    # put legend outside if few clusters, else skip
    if n_clusters <= 30:
        ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            fontsize=7,
            markerscale=3,
        )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot → {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_cluster_sizes(
    labels: np.ndarray,
    *,
    title: str = "Cluster Size Distribution",
    save_path: Path | None = None,
):
    """Bar chart of cluster sizes."""
    unique, counts = np.unique(labels, return_counts=True)
    order = np.argsort(-counts)
    unique, counts = unique[order], counts[order]

    fig, ax = plt.subplots(figsize=(max(8, len(unique) * 0.35), 5))
    ax.bar(range(len(unique)), counts, tick_label=unique)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("# tokens")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot → {save_path}")
    else:
        plt.show()
    plt.close(fig)


def find_best_k(
    vecs: np.ndarray,
    k_range: range = range(5, 105, 5),
    sample_size: int = 10000,
    save_path: Path | None = None,
):
    """Elbow (inertia) + Silhouette analysis to suggest optimal k."""
    # subsample for silhouette (which is O(n^2))
    rng = np.random.default_rng(42)
    if len(vecs) > sample_size:
        idx = rng.choice(len(vecs), size=sample_size, replace=False)
        vecs_sample = vecs[idx]
    else:
        vecs_sample = vecs
        idx = np.arange(len(vecs))

    inertias = []
    silhouettes = []
    ks = list(k_range)

    for k in ks:
        print(f"  k={k} ...", end=" ", flush=True)
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(vecs)
        inertias.append(km.inertia_)
        # silhouette on subsample
        labels_sample = km.labels_[idx]
        sil = silhouette_score(
            vecs_sample, labels_sample, sample_size=min(5000, len(vecs_sample))
        )
        silhouettes.append(sil)
        print(f"inertia={km.inertia_:.1f}  silhouette={sil:.4f}")

    best_k = ks[np.argmax(silhouettes)]
    print(f"\n  Best k by silhouette: {best_k} (score={max(silhouettes):.4f})")

    # plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(ks, inertias, "o-")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Inertia (SSE)")
    ax1.set_title("Elbow Method")
    ax1.grid(True, alpha=0.3)

    ax2.plot(ks, silhouettes, "o-", color="orange")
    ax2.axvline(best_k, ls="--", color="red", alpha=0.7, label=f"best k={best_k}")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Analysis")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Optimal k Selection", fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot → {save_path}")
    else:
        plt.show()
    plt.close(fig)

    return best_k


def save_cluster_report(
    labels: np.ndarray,
    token_strs: list[str],
    save_path: Path,
    max_tokens_per_cluster: int = 50,
):
    """Write a text file listing tokens per cluster."""
    unique_labels = np.unique(labels)
    with open(save_path, "w", encoding="utf-8") as f:
        for lbl in sorted(unique_labels):
            idxs = np.where(labels == lbl)[0]
            header = f"=== Cluster {lbl}  ({len(idxs)} tokens) ==="
            f.write(header + "\n")
            sample = idxs[:max_tokens_per_cluster]
            tokens = [repr(token_strs[i]) for i in sample]
            f.write(", ".join(tokens))
            if len(idxs) > max_tokens_per_cluster:
                f.write(f"  ... and {len(idxs) - max_tokens_per_cluster} more")
            f.write("\n\n")
    print(f"Saved cluster report → {save_path}")


# ---------- CLI --------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Cluster & visualize GPT-2 embeddings")

    p.add_argument(
        "--blackbox-model-name",
        type=str,
        default="gpt2",
        help="HuggingFace model name (used for tokenizer)",
    )
    p.add_argument(
        "--blackbox-model-dir",
        type=str,
        default=r"/scratch/b5bq/pu22650.b5bq/VASAE_out/BlackBoxModels/gpt2",
        help="Directory containing emb.pth / unemb.pth",
    )
    p.add_argument(
        "--method",
        type=str,
        default="kmeans",
        choices=["kmeans", "agglomerative", "dbscan"],
    )
    p.add_argument("--n-clusters", type=int, default=20)
    p.add_argument("--eps", type=float, default=0.5, help="DBSCAN eps")
    p.add_argument("--min-samples", type=int, default=5, help="DBSCAN min_samples")
    p.add_argument(
        "--dim-reduction",
        type=str,
        default="tsne",
        choices=["tsne", "pca"],
    )
    p.add_argument(
        "--normalize-vecs",
        action="store_true",
        help="L2-normalize embeddings before clustering (cluster by direction)",
    )
    p.add_argument(
        "--annotate-n",
        type=int,
        default=3,
        help="Number of token labels to annotate per cluster",
    )
    p.add_argument(
        "--find-best-k",
        action="store_true",
        help="Run elbow + silhouette analysis to find optimal k (then cluster with that k)",
    )
    p.add_argument("--k-min", type=int, default=5, help="Min k for --find-best-k sweep")
    p.add_argument(
        "--k-max", type=int, default=100, help="Max k for --find-best-k sweep"
    )
    p.add_argument("--k-step", type=int, default=5, help="Step for --find-best-k sweep")
    p.add_argument(
        "--output-dir",
        type=str,
        default="outputs/emb_clusters",
    )

    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load embeddings ---------------------------------------------------
    cfg = BlackBoxModelConfig(
        name=args.blackbox_model_name,
        dir=Path(args.blackbox_model_dir) if args.blackbox_model_dir else None,
    )
    print(f"Loading embedding layer from {cfg.dir} ...")
    vecs = get_embeddings(cfg)  # (vocab_size, dim)
    print(f"Embedding matrix shape: {vecs.shape}")

    if args.normalize_vecs:
        vecs = normalize(vecs, norm="l2", axis=1)
        print("Vectors L2-normalised (clustering by direction).")

    # ---- tokenizer for labels ----------------------------------------------
    tokenizer = get_tokenizer(args.blackbox_model_name)
    token_strs = [tokenizer.decode([i]) for i in range(vecs.shape[0])]

    # ---- find best k (optional) --------------------------------------------
    if args.find_best_k:
        print(
            f"Searching best k in [{args.k_min}, {args.k_max}] step {args.k_step} ..."
        )
        best_k = find_best_k(
            vecs,
            k_range=range(args.k_min, args.k_max + 1, args.k_step),
            save_path=out_dir / "optimal_k.png",
        )
        args.n_clusters = best_k
        print(f"Using k={best_k} for clustering.\n")

    # ---- cluster -----------------------------------------------------------
    print(f"Clustering with method={args.method}, n_clusters={args.n_clusters} ...")
    labels = cluster_embeddings(
        vecs,
        method=args.method,
        n_clusters=args.n_clusters,
        eps=args.eps,
        min_samples=args.min_samples,
    )
    n_found = len(set(labels) - {-1})
    print(f"Found {n_found} clusters.")

    # ---- dimensionality reduction ------------------------------------------
    print(f"Reducing to 2-D with {args.dim_reduction} ...")
    coords_2d = reduce_dims(vecs, method=args.dim_reduction)

    # ---- plots -------------------------------------------------------------
    tag = f"{args.method}_k{args.n_clusters}_{args.dim_reduction}"

    plot_clusters_2d(
        coords_2d,
        labels,
        title=f"GPT-2 Embedding Clusters ({args.method}, k={args.n_clusters})",
        save_path=out_dir / f"clusters_{tag}.png",
        annotate_n=args.annotate_n,
        token_strs=token_strs,
    )

    plot_cluster_sizes(
        labels,
        title=f"Cluster sizes ({args.method}, k={args.n_clusters})",
        save_path=out_dir / f"cluster_sizes_{tag}.png",
    )

    save_cluster_report(
        labels,
        token_strs,
        save_path=out_dir / f"cluster_report_{tag}.txt",
    )

    print("Done.")


if __name__ == "__main__":
    main()
