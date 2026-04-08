import json
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings(
    "ignore",
    message="The SAMME.R algorithm",
    category=FutureWarning,
)


RANDOM_STATE_TRAIN = 42
RANDOM_STATE_TEST = 2024
OUTPUT_DIR = Path("visualizations")


def make_moons_3d(n_samples=500, noise=0.1, random_state=None):
    """Replicate the generator from 3D数据集.py.

    The original function creates n_samples points for each class,
    so the total sample count is 2 * n_samples.
    """
    rng = np.random.default_rng(random_state)
    t = np.linspace(0, 2 * np.pi, n_samples)
    x = 1.5 * np.cos(t)
    y = np.sin(t)
    z = np.sin(2 * t)

    x_all = np.vstack(
        [
            np.column_stack([x, y, z]),
            np.column_stack([-x, y - 1, -z]),
        ]
    )
    y_all = np.hstack([np.zeros(n_samples, dtype=int), np.ones(n_samples, dtype=int)])
    x_all += rng.normal(scale=noise, size=x_all.shape)
    return x_all, y_all


@dataclass
class ModelResult:
    name: str
    best_params: dict
    cv_accuracy: float
    test_accuracy: float
    test_f1_macro: float
    y_pred: list
    confusion_matrix: list
    report: dict


def evaluate_model(name, estimator, param_grid, x_train, y_train, x_test, y_test, cv):
    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=1,
    )
    search.fit(x_train, y_train)
    best_model = search.best_estimator_
    y_pred = best_model.predict(x_test)

    return ModelResult(
        name=name,
        best_params=search.best_params_,
        cv_accuracy=float(search.best_score_),
        test_accuracy=float(accuracy_score(y_test, y_pred)),
        test_f1_macro=float(f1_score(y_test, y_pred, average="macro")),
        y_pred=y_pred.tolist(),
        confusion_matrix=confusion_matrix(y_test, y_pred).tolist(),
        report=classification_report(y_test, y_pred, output_dict=True, zero_division=0),
    )


def sanitize_filename(name):
    return (
        name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("+", "plus")
    )


def plot_dataset(train_x, train_y, test_x, test_y):
    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(121, projection="3d")
    scatter1 = ax1.scatter(
        train_x[:, 0],
        train_x[:, 1],
        train_x[:, 2],
        c=train_y,
        cmap="viridis",
        s=18,
        alpha=0.8,
    )
    ax1.set_title("Training Set (1000 samples)")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.legend(*scatter1.legend_elements(), title="Class", loc="upper right")

    ax2 = fig.add_subplot(122, projection="3d")
    scatter2 = ax2.scatter(
        test_x[:, 0],
        test_x[:, 1],
        test_x[:, 2],
        c=test_y,
        cmap="viridis",
        s=20,
        alpha=0.85,
    )
    ax2.set_title("Test Set (500 samples)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.legend(*scatter2.legend_elements(), title="Class", loc="upper right")

    fig.suptitle("3D Data Distribution", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "dataset_distribution.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_metric_comparison(results):
    names = [result.name for result in results]
    cv_scores = [result.cv_accuracy for result in results]
    test_scores = [result.test_accuracy for result in results]
    f1_scores = [result.test_f1_macro for result in results]

    x = np.arange(len(names))
    width = 0.24

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, cv_scores, width, label="CV Accuracy", color="#4C78A8")
    bars2 = ax.bar(x, test_scores, width, label="Test Accuracy", color="#F58518")
    bars3 = ax.bar(x + width, f1_scores, width, label="Macro F1", color="#54A24B")

    ax.set_title("Model Performance Comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0.6, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.005,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "model_metric_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(results):
    cols = 3
    rows = int(np.ceil(len(results) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4.8 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, result in zip(axes, results):
        cm = np.array(result.confusion_matrix)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(result.name)
        ax.set_xticks([0, 1], labels=["Pred C0", "Pred C1"])
        ax.set_yticks([0, 1], labels=["True C0", "True C1"])

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=11,
                )

        ax.set_xlabel(f"Acc={result.test_accuracy:.3f}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[len(results):]:
        ax.axis("off")

    fig.suptitle("Confusion Matrices of All Models", fontsize=15)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrices_overview.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    for result in results:
        cm = np.array(result.confusion_matrix)
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(result.name)
        ax.set_xticks([0, 1], labels=["Pred C0", "Pred C1"])
        ax.set_yticks([0, 1], labels=["True C0", "True C1"])

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=12,
                )

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(
            OUTPUT_DIR / f"confusion_matrix_{sanitize_filename(result.name)}.png",
            dpi=220,
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_prediction_comparison(test_x, y_test, results):
    top_results = results[:3]
    fig = plt.figure(figsize=(16, 5))

    for idx, result in enumerate(top_results, start=1):
        ax = fig.add_subplot(1, 3, idx, projection="3d")
        y_pred = np.array(result.y_pred)
        correct = y_pred == y_test
        colors = np.where(correct, "#2ca02c", "#d62728")

        ax.scatter(test_x[:, 0], test_x[:, 1], test_x[:, 2], c=colors, s=18, alpha=0.85)
        ax.set_title(f"{result.name}\nGreen=Correct, Red=Wrong")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    fig.suptitle("Prediction Quality on Test Set (Top 3 Models)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "top3_prediction_quality.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    # The assignment asks for 1000 training points total and 500 testing points total.
    # Because the provided generator creates n_samples per class, we use 500 and 250 here.
    x_train, y_train = make_moons_3d(n_samples=500, noise=0.2, random_state=RANDOM_STATE_TRAIN)
    x_test, y_test = make_moons_3d(n_samples=250, noise=0.2, random_state=RANDOM_STATE_TEST)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

    models = [
        (
            "Decision Tree",
            DecisionTreeClassifier(random_state=123),
            {
                "criterion": ["gini", "entropy"],
                "max_depth": [None, 3, 5, 7, 10],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
        ),
        (
            "AdaBoost + Decision Tree",
            AdaBoostClassifier(
                estimator=DecisionTreeClassifier(random_state=123),
                random_state=123,
            ),
            {
                "estimator__max_depth": [1, 2, 3],
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.3, 0.5, 1.0],
            },
        ),
        (
            "SVM (linear)",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("svc", SVC(kernel="linear")),
                ]
            ),
            {
                "svc__C": [0.1, 1, 10, 100],
            },
        ),
        (
            "SVM (poly)",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("svc", SVC(kernel="poly")),
                ]
            ),
            {
                "svc__C": [0.1, 1, 10],
                "svc__degree": [2, 3, 4],
                "svc__gamma": ["scale", "auto"],
                "svc__coef0": [0.0, 1.0],
            },
        ),
        (
            "SVM (rbf)",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("svc", SVC(kernel="rbf")),
                ]
            ),
            {
                "svc__C": [0.1, 1, 10, 100],
                "svc__gamma": ["scale", "auto", 0.1, 1.0],
            },
        ),
        (
            "SVM (sigmoid)",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("svc", SVC(kernel="sigmoid")),
                ]
            ),
            {
                "svc__C": [0.1, 1, 10, 100],
                "svc__gamma": ["scale", "auto", 0.1, 1.0],
                "svc__coef0": [-1.0, 0.0, 1.0],
            },
        ),
    ]

    results = []
    for name, estimator, param_grid in models:
        result = evaluate_model(name, estimator, param_grid, x_train, y_train, x_test, y_test, cv)
        results.append(result)

    results.sort(key=lambda item: item.test_accuracy, reverse=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    plot_dataset(x_train, y_train, x_test, y_test)
    plot_metric_comparison(results)
    plot_confusion_matrices(results)
    plot_prediction_comparison(x_test, y_test, results)

    print("Training set size:", len(y_train), "(C0 =", int((y_train == 0).sum()), ", C1 =", int((y_train == 1).sum()), ")")
    print("Test set size:", len(y_test), "(C0 =", int((y_test == 0).sum()), ", C1 =", int((y_test == 1).sum()), ")")
    print()
    print("Model comparison (sorted by test accuracy)")
    print("-" * 88)
    print(f"{'Model':<24} {'CV Acc':>10} {'Test Acc':>10} {'Macro F1':>10}")
    print("-" * 88)
    for result in results:
        print(
            f"{result.name:<24} "
            f"{result.cv_accuracy:>10.4f} "
            f"{result.test_accuracy:>10.4f} "
            f"{result.test_f1_macro:>10.4f}"
        )
    print("-" * 88)
    print()

    for result in results:
        print(f"[{result.name}]")
        print("Best params:", json.dumps(result.best_params, ensure_ascii=False))
        print("Confusion matrix:", result.confusion_matrix)
        print(
            "Class-wise F1:",
            {
                label: round(result.report[label]["f1-score"], 4)
                for label in ["0", "1"]
            },
        )
        print()

    payload = {
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "results": [asdict(result) for result in results],
    }
    with open("evaluation_results.json", "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
