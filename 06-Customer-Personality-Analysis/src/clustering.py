"""clustering.py — Module 6: Customer Segmentation.

Reusable clustering pipeline built on top of the Module 5 engineered
dataset. Imports feature_engineering.validate_dataset() instead of
duplicating validation logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from feature_engineering import FEATURES_PATH, validate_dataset as _validate_features

PROJECT_DIR = Path.cwd()
OUTPUTS_DIR = PROJECT_DIR / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"

NON_CLUSTERING_COLS = ["Complain", "Response", "AcceptedCmp1", "AcceptedCmp2",
                        "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5"]


def load_feature_dataset(path: Path = FEATURES_PATH) -> pd.DataFrame:
    """Load the Module 5 engineered dataset."""
    return pd.read_csv(path)


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the dataset via the Module 5 validation logic."""
    return _validate_features(df)


def select_features(df: pd.DataFrame, exclude: list[str] | None = None) -> pd.DataFrame:
    """Select numeric columns suitable for clustering, dropping flags by default."""
    exclude = NON_CLUSTERING_COLS if exclude is None else exclude
    cols = [c for c in df.select_dtypes(include="number").columns if c not in exclude]
    return df[cols].copy()


def apply_standard_scaler(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """Standardize features to zero mean and unit variance."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    return pd.DataFrame(scaled, columns=df.columns, index=df.index), scaler


def apply_minmax_scaler(df: pd.DataFrame) -> tuple[pd.DataFrame, MinMaxScaler]:
    """Scale features to the [0, 1] range."""
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    return pd.DataFrame(scaled, columns=df.columns, index=df.index), scaler


def save_scaler(scaler: StandardScaler | MinMaxScaler, path: Path) -> Path:
    """Persist a fitted scaler to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)
    return path


def elbow_method(X: pd.DataFrame, k_range: range = range(2, 11),
                  save_path: Path | None = None) -> pd.DataFrame:
    """Compute KMeans inertia across a range of k and optionally plot the elbow curve."""
    inertias = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(X).inertia_ for k in k_range]
    result = pd.DataFrame({"k": list(k_range), "inertia": inertias})
    if save_path is not None:
        fig, ax = plt.subplots()
        ax.plot(result["k"], result["inertia"], marker="o")
        ax.set_xlabel("k")
        ax.set_ylabel("Inertia")
        ax.set_title("Elbow Method")
        _save_fig(fig, save_path)
    return result


def silhouette_analysis(X: pd.DataFrame, k_range: range = range(2, 11),
                         save_path: Path | None = None) -> pd.DataFrame:
    """Compute silhouette scores across a range of k and optionally plot them."""
    scores = [silhouette_score(X, KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X))
              for k in k_range]
    result = pd.DataFrame({"k": list(k_range), "silhouette_score": scores})
    if save_path is not None:
        fig, ax = plt.subplots()
        ax.plot(result["k"], result["silhouette_score"], marker="o")
        ax.set_xlabel("k")
        ax.set_ylabel("Silhouette Score")
        ax.set_title("Silhouette Analysis")
        _save_fig(fig, save_path)
    return result


def davies_bouldin_analysis(X: pd.DataFrame, k_range: range = range(2, 11)) -> pd.DataFrame:
    """Compute Davies-Bouldin scores across a range of k (lower is better)."""
    scores = [davies_bouldin_score(X, KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X))
              for k in k_range]
    return pd.DataFrame({"k": list(k_range), "davies_bouldin_score": scores})


def train_kmeans(X: pd.DataFrame, n_clusters: int) -> tuple[KMeans, np.ndarray]:
    """Fit a KMeans model and return the model and cluster labels."""
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(X)
    return model, model.labels_


def train_hierarchical(X: pd.DataFrame, n_clusters: int,
                        save_path: Path | None = None) -> tuple[AgglomerativeClustering, np.ndarray]:
    """Fit Agglomerative Clustering and optionally save a dendrogram."""
    model = AgglomerativeClustering(n_clusters=n_clusters).fit(X)
    if save_path is not None:
        Z = linkage(X, method="ward")
        fig, ax = plt.subplots(figsize=(10, 6))
        dendrogram(Z, truncate_mode="lastp", p=30, ax=ax)
        ax.set_title("Hierarchical Clustering Dendrogram")
        _save_fig(fig, save_path)
    return model, model.labels_


def train_gmm(X: pd.DataFrame, n_components: int) -> tuple[GaussianMixture, np.ndarray]:
    """Fit a Gaussian Mixture Model and return the model and cluster labels."""
    model = GaussianMixture(n_components=n_components, random_state=42).fit(X)
    return model, model.predict(X)


def train_dbscan(X: pd.DataFrame, eps: float = 0.5, min_samples: int = 5) -> tuple[DBSCAN, np.ndarray]:
    """Fit DBSCAN and return the model and cluster labels."""
    model = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    return model, model.labels_


def compare_models(X: pd.DataFrame, labels_by_model: dict[str, np.ndarray]) -> pd.DataFrame:
    """Compare clustering algorithms using silhouette and Davies-Bouldin scores."""
    records = []
    for name, labels in labels_by_model.items():
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        valid = n_clusters > 1
        records.append({
            "model": name,
            "n_clusters": n_clusters,
            "silhouette_score": silhouette_score(X, labels) if valid else np.nan,
            "davies_bouldin_score": davies_bouldin_score(X, labels) if valid else np.nan,
        })
    return pd.DataFrame(records)


def profile_clusters(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Compute per-cluster mean feature values."""
    profiled = df.copy()
    profiled["Cluster"] = labels
    return profiled.groupby("Cluster").mean()


def visualize_pca(X: pd.DataFrame, labels: np.ndarray, save_path: Path | None = None) -> pd.DataFrame:
    """Project features to 2D with PCA and plot clusters."""
    components = PCA(n_components=2, random_state=42).fit_transform(X)
    result = pd.DataFrame(components, columns=["PC1", "PC2"])
    result["Cluster"] = labels
    if save_path is not None:
        fig, ax = plt.subplots()
        sns.scatterplot(data=result, x="PC1", y="PC2", hue="Cluster", palette="tab10", ax=ax)
        ax.set_title("PCA Cluster Projection")
        _save_fig(fig, save_path)
    return result


def visualize_tsne(X: pd.DataFrame, labels: np.ndarray, save_path: Path | None = None) -> pd.DataFrame:
    """Project features to 2D with t-SNE and plot clusters."""
    components = TSNE(n_components=2, random_state=42, init="pca").fit_transform(X)
    result = pd.DataFrame(components, columns=["Dim1", "Dim2"])
    result["Cluster"] = labels
    if save_path is not None:
        fig, ax = plt.subplots()
        sns.scatterplot(data=result, x="Dim1", y="Dim2", hue="Cluster", palette="tab10", ax=ax)
        ax.set_title("t-SNE Cluster Projection")
        _save_fig(fig, save_path)
    return result


def cluster_heatmap(profile: pd.DataFrame, save_path: Path | None = None) -> None:
    """Plot a heatmap of cluster profile means (z-normalized per feature)."""
    normalized = (profile - profile.mean()) / profile.std()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(normalized.T, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Cluster Profile Heatmap (z-normalized)")
    if save_path is not None:
        _save_fig(fig, save_path)


def radar_chart(profile: pd.DataFrame, features: list[str], save_path: Path | None = None) -> None:
    """Plot a radar chart comparing clusters across selected features."""
    normalized = (profile[features] - profile[features].min()) / (
        profile[features].max() - profile[features].min())
    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    for cluster_id, row in normalized.iterrows():
        values = row.tolist() + row.tolist()[:1]
        ax.plot(angles, values, label=f"Cluster {cluster_id}")
        ax.fill(angles, values, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features)
    ax.set_title("Cluster Radar Chart")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    if save_path is not None:
        _save_fig(fig, save_path)


def stability_analysis(X: pd.DataFrame, n_clusters: int, n_runs: int = 5) -> pd.DataFrame:
    """Assess KMeans stability across random seeds via pairwise Adjusted Rand Index."""
    runs = [KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(X)
            for seed in range(n_runs)]
    scores = [adjusted_rand_score(runs[i], runs[j])
              for i in range(n_runs) for j in range(i + 1, n_runs)]
    return pd.DataFrame({"pairwise_ari": scores, "mean_ari": np.mean(scores)})


def save_final_pipeline(scaler: StandardScaler | MinMaxScaler, model, feature_columns: list[str],
                         path: Path = MODELS_DIR / "pipeline.pkl") -> Path:
    """Save the fitted scaler, model, and feature list as a single deployable pipeline object."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "model": model, "features": feature_columns}, path)
    return path


def save_model(model, path: Path) -> Path:
    """Persist a fitted model to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def generate_cluster_summary(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Summarize cluster sizes and their share of the customer base."""
    summary = pd.Series(labels).value_counts().sort_index().rename("Count").reset_index()
    summary.columns = ["Cluster", "Count"]
    summary["Percentage"] = (summary["Count"] / len(labels) * 100).round(2)
    return summary


def export_reports(reports: dict[str, pd.DataFrame], reports_dir: Path = REPORTS_DIR) -> list[Path]:
    """Export each named DataFrame to its own Excel file in reports_dir."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, data in reports.items():
        path = reports_dir / f"{name}.xlsx"
        data.to_excel(path, index=True)
        paths.append(path)
    return paths


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
