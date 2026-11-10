# ML Playground

An interactive, educational machine learning lab for exploring regression, classification, and clustering. Change the dataset, model, and training controls, then inspect the data visualization, metrics, and train/test gap.

## Features

- Regression: linear regression and decision tree regression.
- Classification: K-nearest neighbors and decision tree classification.
- Clustering: K-means with adjustable cluster count.
- Reproducible train/test splits, optional feature scaling, and configurable tree complexity.
- Visualizations, metric hover tips, and a task-aware popover explaining every result in Spanish.

## Run locally

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## How to use

Choose a task and dataset in the sidebar, select a model, then adjust its controls. The app uses a fixed random seed so experiments are repeatable. Classification and regression use a stratified (classification) or ordinary (regression) holdout split. Scaling is fitted only on training data to avoid leakage. Clustering is unsupervised and therefore has no train/test score.

## Metrics

- **Classification:** accuracy, precision, recall, F1, and confusion matrix.
- **Regression:** MAE, RMSE, and R².
- **Clustering:** silhouette score (when there is more than one cluster and fewer clusters than samples).

The train score beside the test score helps illustrate overfitting. A high training score with a substantially lower test score can indicate that a model has learned patterns specific to its training examples.

## Demo walkthrough

Run the app locally and open `http://localhost:8501`. Try this short walkthrough:

1. Use **Classification → Iris → K-nearest neighbors** and change `k` to see the scores respond.
2. Switch to **Regression → Synthetic** and compare MAE, RMSE, and R².
3. Switch to **Clustering → Synthetic blobs** and change the requested number of clusters.
4. Open **¿Qué significa cada resultado?** for plain-language explanations, or hover over a metric's help icon for a concise definition.

For classification, the app also shows the mapping from numeric class IDs to their names. In the confusion matrix, rows are actual classes and columns are predicted classes.

## Run the automated checks

```bash
python -m unittest discover -s tests -v
```

The standard-library test suite covers the bundled datasets, reproducibility, supported estimators, metric outputs, train-only scaling, and plotting helpers.

## Project structure

```text
app.py          Streamlit interface and experiment flow
src/ml_playground/  Dataset, model, metric, and plot helpers
```

## Limitations

This is a learning tool using small built-in scikit-learn datasets. Metrics depend on the selected dataset and split; they are not guarantees of real-world performance. The two-feature view is a projection for visualization, while metrics use all selected features.

## License

MIT. See [LICENSE](LICENSE).
