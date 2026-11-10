"""Automated checks for the experiment pipeline and its supported datasets."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ml_playground.experiments import (  # noqa: E402
    calculate_metrics,
    load_dataset,
    plot_confusion,
    plot_data,
    train_model,
)


class DatasetTests(unittest.TestCase):
    def test_every_dataset_has_expected_shape_and_finite_values(self):
        datasets = [
            ("Classification", "Iris", 300, 150),
            ("Classification", "Breast cancer", 300, 569),
            ("Regression", "Synthetic", 300, 300),
            ("Regression", "Diabetes", 300, 442),
            ("Clustering", "Synthetic blobs", 240, 240),
        ]
        for task, name, samples, expected_rows in datasets:
            with self.subTest(task=task, dataset=name):
                X, y = load_dataset(task, name, samples)
                self.assertEqual(len(X), expected_rows)
                self.assertTrue(np.isfinite(X.to_numpy()).all())
                if task == "Clustering":
                    self.assertIsNone(y)
                    self.assertEqual(X.shape[1], 2)
                else:
                    self.assertEqual(len(X), len(y))
                    self.assertTrue(np.isfinite(y.to_numpy()).all())

    def test_synthetic_regression_and_clustering_are_reproducible(self):
        for task, dataset in [("Regression", "Synthetic"), ("Clustering", "Synthetic blobs")]:
            with self.subTest(task=task):
                X1, y1 = load_dataset(task, dataset, 120)
                X2, y2 = load_dataset(task, dataset, 120)
                np.testing.assert_array_equal(X1.to_numpy(), X2.to_numpy())
                if y1 is not None:
                    np.testing.assert_array_equal(y1.to_numpy(), y2.to_numpy())


class ClassificationTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y = load_dataset("Classification", "Iris")

    def test_knn_pipeline_returns_valid_metrics_and_stratified_split(self):
        result = train_model("Classification", "K-nearest neighbors", self.X, self.y,
                             test_size=.25, scale=True, neighbors=7)
        metrics = calculate_metrics("Classification", result)
        self.assertEqual(len(result["y_test"]), 38)
        self.assertEqual(len(result["y_train"]), 112)
        self.assertEqual(set(metrics), {"Accuracy", "Precision (macro)", "Recall (macro)",
                                        "F1 (macro)", "Train accuracy", "Confusion matrix"})
        self.assertEqual(metrics["Confusion matrix"].sum(), len(result["y_test"]))
        for key in ("Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)", "Train accuracy"):
            self.assertGreaterEqual(metrics[key], 0)
            self.assertLessEqual(metrics[key], 1)

    def test_decision_tree_classification_runs_without_scaling(self):
        result = train_model("Classification", "Decision tree", self.X, self.y,
                             test_size=.2, scale=False, max_depth=3)
        metrics = calculate_metrics("Classification", result)
        self.assertEqual(metrics["Confusion matrix"].shape, (3, 3))


class RegressionTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y = load_dataset("Regression", "Synthetic", samples=180)

    def test_linear_regression_returns_finite_regression_metrics(self):
        result = train_model("Regression", "Linear regression", self.X, self.y,
                             test_size=.2, scale=True)
        metrics = calculate_metrics("Regression", result)
        self.assertEqual(set(metrics), {"MAE", "RMSE", "R²", "Train R²"})
        self.assertGreaterEqual(metrics["MAE"], 0)
        self.assertGreaterEqual(metrics["RMSE"], 0)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))

    def test_tree_regression_runs_without_scaling(self):
        result = train_model("Regression", "Decision tree", self.X, self.y,
                             test_size=.25, scale=False, max_depth=3)
        self.assertTrue(np.isfinite(calculate_metrics("Regression", result)["R²"]))

    def test_scaler_is_fitted_on_training_partition_only(self):
        result = train_model("Regression", "Linear regression", self.X, self.y,
                             test_size=.25, scale=True)
        fitted_scaler = result["model"].named_steps["standardscaler"]
        np.testing.assert_allclose(fitted_scaler.mean_, result["X_train"].mean(axis=0).to_numpy())


class ClusteringTests(unittest.TestCase):
    def test_kmeans_returns_requested_number_of_clusters_and_silhouette(self):
        X, _ = load_dataset("Clustering", "Synthetic blobs", samples=180)
        result = train_model("Clustering", "K-means", X, None, .25, False, clusters=4)
        self.assertEqual(len(result["labels"]), len(X))
        self.assertEqual(len(np.unique(result["labels"])), 4)


class VisualizationTests(unittest.TestCase):
    def test_dataset_plot_handles_both_two_and_many_features(self):
        for task, name in [("Classification", "Iris"), ("Clustering", "Synthetic blobs")]:
            X, y = load_dataset(task, name, 120)
            with self.subTest(dataset=name):
                labels = None
                if task == "Clustering":
                    labels = train_model(task, "K-means", X, y, .25, False)["labels"]
                figure = plot_data(X, y, task, labels)
                self.assertEqual(len(figure.axes), 2)  # scatter plot and colorbar

    def test_confusion_matrix_plot_has_one_cell_per_class_pair(self):
        figure = plot_confusion(np.array([[8, 1], [2, 9]]))
        self.assertEqual(len(figure.axes), 2)
        self.assertEqual(len(figure.axes[0].texts), 4)


if __name__ == "__main__":
    unittest.main()
