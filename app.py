from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from ml_playground.experiments import (  # noqa: E402
    calculate_metrics, load_dataset, plot_confusion, plot_data, train_model,
)

st.set_page_config(page_title="ML Playground", page_icon="🧪", layout="wide")
st.title("🧪 ML Playground")
st.caption("Experiment with classic machine learning models and see how your choices change the results.")

with st.sidebar:
    st.header("Experiment setup")
    task = st.selectbox("Task", ["Classification", "Regression", "Clustering"])
    datasets = {"Classification": ["Iris", "Breast cancer"],
                "Regression": ["Synthetic", "Diabetes"],
                "Clustering": ["Synthetic blobs"]}
    dataset = st.selectbox("Dataset", datasets[task], help="Dataset used for this experiment. The built-in datasets are small examples for learning.")
    algorithms = {"Classification": ["K-nearest neighbors", "Decision tree"],
                  "Regression": ["Linear regression", "Decision tree"],
                  "Clustering": ["K-means"]}
    algorithm = st.selectbox("Model", algorithms[task], help="Algorithm that will learn from the selected dataset.")
    st.divider()
    if task != "Clustering":
        test_size = st.slider("Test set (%)", 10, 40, 25, 5,
                              help="Share of labeled examples held out to evaluate the model after training.") / 100
        scale = st.checkbox("Scale features", value=(algorithm == "K-nearest neighbors"),
                            help="Standardize each feature using training-set statistics. Useful for distance-based KNN.")
        if task == "Classification" and algorithm == "K-nearest neighbors":
            neighbors = st.slider("Neighbors (k)", 1, 25, 7, 2,
                                  help="Number of nearby training examples that vote on a class. Smaller k is more sensitive to individual points.")
        else:
            neighbors = 7
        if algorithm == "Decision tree":
            max_depth = st.slider("Maximum tree depth", 1, 20, 4,
                                  help="Maximum number of decision levels in the tree. Deeper trees can fit more complex patterns and may overfit.")
        else:
            max_depth = 4
        clusters = 3
    else:
        samples = st.slider("Number of samples", 100, 800, 300, 50,
                            help="How many synthetic points to generate for the clustering experiment.")
        clusters = st.slider("Number of clusters", 2, 8, 3,
                             help="How many groups K-means should find. This is supplied by you, not inferred automatically.")
        test_size, scale, neighbors, max_depth = .25, False, 7, 4

X, y = load_dataset(task, dataset, locals().get("samples", 300))
result = train_model(task, algorithm, X, y, test_size, scale, neighbors, max_depth, clusters)

left, right = st.columns([1, 1])
with left:
    st.subheader("Dataset")
    st.write(f"**{len(X):,} samples** · **{X.shape[1]} features**" +
             (f" · **{y.nunique()} classes**" if task == "Classification" else ""))
    st.pyplot(plot_data(X, y, task, result.get("labels")), use_container_width=True)
    if task == "Classification":
        class_ids = {"Iris": "0 = setosa · 1 = versicolor · 2 = virginica",
                     "Breast cancer": "0 = malignant · 1 = benign"}
        st.caption(f"Class IDs: {class_ids[dataset]}")
with right:
    st.subheader("Results")
    if task == "Clustering":
        from sklearn.metrics import silhouette_score
        labels = result["labels"]
        score = silhouette_score(X, labels) if len(set(labels)) > 1 and len(set(labels)) < len(X) else float("nan")
        st.metric("Silhouette score", f"{score:.3f}",
                  help="Measures whether points are close to their own cluster and far from other clusters. Range: −1 to 1; higher is usually better.")
        st.caption("Higher is better (range −1 to 1). It measures how well separated the discovered groups are.")
    else:
        metrics = calculate_metrics(task, result)
        names = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"] if task == "Classification" else ["MAE", "RMSE", "R²"]
        cols = st.columns(len(names))
        metric_help = {
            "Accuracy": "Share of test examples classified correctly. Higher is better, but class imbalance can make it misleading.",
            "Precision (macro)": "Of the examples predicted as each class, how many were correct. Macro means each class counts equally; higher is better.",
            "Recall (macro)": "Of the real examples in each class, how many the model found. Macro means each class counts equally; higher is better.",
            "F1 (macro)": "Balance between precision and recall, averaged equally across classes. Higher is better.",
            "MAE": "Mean absolute error: the average size of the prediction error in the target's units. Lower is better.",
            "RMSE": "Root mean squared error: like MAE, but large errors count more strongly. Lower is better.",
            "R²": "How much target variation the model explains on the test set. 1 is perfect, 0 matches predicting the mean, and a negative value is worse.",
        }
        for col, name in zip(cols, names):
            col.metric(name, f"{metrics[name]:.3f}", help=metric_help[name])
        train_name, test_name = ("Train accuracy", "Accuracy") if task == "Classification" else ("Train R²", "R²")
        train_score, test_score = metrics[train_name], metrics[test_name]
        gap = train_score - test_score
        st.info(f"**Train vs. test:** {train_score:.3f} vs. {test_score:.3f}. " +
                ("The gap may indicate overfitting." if gap > .15 else "The scores are relatively close on this split."))
        st.caption("Training score: fit on examples the model learned from. Test score: evaluated on examples held out during training. A large gap can signal overfitting.")
        st.caption("The test set is held out during training. Compare metrics across settings; one split is not a guarantee of future performance.")
        if task == "Classification":
            st.caption("Confusion matrix: rows are the true classes, columns are the model's predictions. The diagonal contains correct predictions; off-diagonal cells show mistakes.")
            st.pyplot(plot_confusion(metrics["Confusion matrix"]), use_container_width=True)

with st.popover("¿Qué significa cada resultado?"):
    if task == "Classification":
        st.markdown("""
        - **Accuracy:** porcentaje de ejemplos de prueba clasificados correctamente.
        - **Precision (macro):** cuando el modelo predice una clase, qué proporción de esas predicciones acierta; calcula cada clase por igual.
        - **Recall (macro):** qué proporción de los ejemplos reales de cada clase encuentra el modelo; calcula cada clase por igual.
        - **F1 (macro):** equilibrio entre precision y recall, con el mismo peso para cada clase.
        - **Train accuracy:** aciertos sobre los ejemplos usados para aprender. Compárala con Accuracy, que se mide en ejemplos reservados.
        - **Matriz de confusión:** filas = clase real; columnas = predicción. La diagonal son aciertos y las demás celdas son errores.

        Para Accuracy, Precision, Recall y F1, valores más altos suelen ser mejores. Precision y Recall ayudan a entender qué tipo de error comete el modelo.
        """)
        st.caption(f"IDs de clase para {dataset}: {class_ids[dataset]}.")
    elif task == "Regression":
        st.markdown("""
        - **MAE:** error absoluto medio, en las mismas unidades que el objetivo. Por ejemplo, MAE = 5 significa que las predicciones se alejan 5 unidades en promedio. Más bajo es mejor.
        - **RMSE:** error cuadrático medio con raíz. También usa las unidades del objetivo y penaliza más los errores grandes. Más bajo es mejor.
        - **R²:** variación del objetivo explicada en los datos de prueba. 1 es perfecto; 0 equivale a predecir el promedio; un valor negativo es peor que esa referencia.
        - **Train R²:** R² calculado con los datos de entrenamiento. Compáralo con R² para detectar si el modelo se ajusta mucho mejor a los ejemplos vistos.
        """)
    else:
        st.markdown("""
        - **Silhouette score:** resume qué tan compacto y separado está cada grupo. Va de −1 a 1: cerca de 1 suele indicar grupos claros, cerca de 0 grupos mezclados, y menor que 0 posibles asignaciones incorrectas. Más alto suele ser mejor.
        - **Número de clusters:** cantidad de grupos que le pedimos encontrar a K-means; el algoritmo no la elige automáticamente.
        - **Colores y números:** cada color representa el grupo asignado a un punto. Los números son identificadores de grupo, no un ranking.
        """)

with st.expander("About this experiment"):
    st.markdown("""
    - **Preprocessing:** scaling, when enabled, is learned from training data only.
    - **Split:** fixed random seed for repeatability; classification keeps class proportions similar in both sets.
    - **Visualization:** datasets with more than two features are projected into two dimensions for the scatter plot. Model metrics use every feature.
    - **Overfitting:** a large train/test gap can mean the model fits training examples better than unseen examples.
    """)
