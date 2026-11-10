"""Dataset, estimator, metric, and visualization helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_iris,
    make_blobs,
    make_regression,
)
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

SEED = 42


def load_dataset(task: str, name: str, samples: int = 300) -> tuple[pd.DataFrame, pd.Series | None]:
    """Return a small built-in dataset as features and an optional target."""
    if task == "Classification":
        loader = {"Iris": load_iris, "Breast cancer": load_breast_cancer}[name]
        data = loader()
        return pd.DataFrame(data.data, columns=data.feature_names), pd.Series(data.target, name="target")
    if task == "Regression":
        if name == "Diabetes":
            data = load_diabetes()
            return pd.DataFrame(data.data, columns=data.feature_names), pd.Series(data.target, name="target")
        X, y = make_regression(n_samples=samples, n_features=2, n_informative=2,
                               noise=22, random_state=SEED)
        return pd.DataFrame(X, columns=["feature_1", "feature_2"]), pd.Series(y, name="target")
    X, _ = make_blobs(n_samples=samples, centers=3, n_features=2,
                      cluster_std=1.5, random_state=SEED)
    return pd.DataFrame(X, columns=["feature_1", "feature_2"]), None


def train_model(task: str, algorithm: str, X: pd.DataFrame, y: pd.Series | None,
                test_size: float, scale: bool, neighbors: int = 7,
                max_depth: int = 4, clusters: int = 3):
    """Fit an estimator and return it with split data and predictions."""
    if task == "Clustering":
        model = KMeans(n_clusters=clusters, n_init=10, random_state=SEED)
        labels = model.fit_predict(X)
        return {"model": model, "labels": labels, "X": X}

    stratify = y if task == "Classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=SEED, stratify=stratify
    )
    if task == "Classification":
        estimator = (KNeighborsClassifier(n_neighbors=neighbors)
                     if algorithm == "K-nearest neighbors"
                     else DecisionTreeClassifier(max_depth=max_depth, random_state=SEED))
    else:
        estimator = (LinearRegression() if algorithm == "Linear regression"
                     else DecisionTreeRegressor(max_depth=max_depth, random_state=SEED))
    model = make_pipeline(StandardScaler(), estimator) if scale else estimator
    model.fit(X_train, y_train)
    return {"model": model, "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
            "train_pred": model.predict(X_train), "test_pred": model.predict(X_test)}


def calculate_metrics(task: str, result: dict) -> dict:
    """Calculate metrics from the held-out test set and training set."""
    if task == "Classification":
        yt, yp = result["y_test"], result["test_pred"]
        return {"Accuracy": accuracy_score(yt, yp),
                "Precision (macro)": precision_score(yt, yp, average="macro", zero_division=0),
                "Recall (macro)": recall_score(yt, yp, average="macro", zero_division=0),
                "F1 (macro)": f1_score(yt, yp, average="macro", zero_division=0),
                "Train accuracy": accuracy_score(result["y_train"], result["train_pred"]),
                "Confusion matrix": confusion_matrix(yt, yp)}
    yt, yp = result["y_test"], result["test_pred"]
    return {"MAE": mean_absolute_error(yt, yp),
            "RMSE": np.sqrt(mean_squared_error(yt, yp)),
            "R²": r2_score(yt, yp),
            "Train R²": r2_score(result["y_train"], result["train_pred"])}


def plot_data(X: pd.DataFrame, y: pd.Series | None, task: str,
              labels: np.ndarray | None = None) -> Figure:
    """Plot a two-dimensional overview (PCA projection for wider data)."""
    if X.shape[1] > 2:
        points = PCA(n_components=2, random_state=SEED).fit_transform(StandardScaler().fit_transform(X))
        axis_names = ("Principal component 1", "Principal component 2")
    else:
        points = X.iloc[:, :2].to_numpy()
        axis_names = (X.columns[0], X.columns[1])
    color = labels if labels is not None else (y.to_numpy() if y is not None else None)
    fig = Figure(figsize=(7, 4.5), tight_layout=True)
    ax = fig.subplots()
    scatter_args = {"c": color, "cmap": "viridis"} if color is not None else {"color": "#5271a5"}
    scatter = ax.scatter(points[:, 0], points[:, 1], **scatter_args, alpha=.8,
                         edgecolors="white", linewidths=.35)
    if color is not None:
        fig.colorbar(scatter, ax=ax, label="Class / cluster" if task != "Regression" else "Target")
    ax.set_xlabel(axis_names[0]); ax.set_ylabel(axis_names[1])
    ax.set_title("Dataset overview" + (" (2D projection)" if X.shape[1] > 2 else ""))
    ax.grid(alpha=.2)
    return fig


def plot_confusion(matrix: np.ndarray) -> Figure:
    fig = Figure(figsize=(5, 4), tight_layout=True)
    ax = fig.subplots()
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(title="Confusion matrix", xlabel="Predicted label", ylabel="True label")
    ax.set_xticks(range(matrix.shape[1])); ax.set_yticks(range(matrix.shape[0]))
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center")
    return fig
