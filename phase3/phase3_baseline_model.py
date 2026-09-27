"""
PHASE 3 — Baseline Models

Project:
    SMS Suraksha — Multi-lingual Indian Fraud SMS Detector

Input:
    ../phase2/train.csv
    ../phase2/val.csv
    ../phase2/test.csv
    ../phase2/tfidf_vectorizer.pkl

Models:
    1. Logistic Regression
    2. Linear SVM
    3. Multinomial Naive Bayes

Important:
    - Validation set is used for model selection.
    - Test set is used only for final evaluation.
    - Translation-safe splits were already created in Phase 2
      using message_id.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.sparse import hstack, csr_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    ConfusionMatrixDisplay,
)


# ==============================================================
# 0. PATHS
# ==============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PHASE2_DIR = os.path.join(
    BASE_DIR,
    "..",
    "phase2"
)

OUT_DIR = os.path.join(
    BASE_DIR,
    "eda_plots"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(
    OUT_DIR,
    exist_ok=True
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ==============================================================
# 1. LOAD DATA
# ==============================================================

TEXT_COL = "text"
LABEL_COL = "label_binary"
LANG_COL = "language"


train_path = os.path.join(
    PHASE2_DIR,
    "train.csv"
)

val_path = os.path.join(
    PHASE2_DIR,
    "val.csv"
)

test_path = os.path.join(
    PHASE2_DIR,
    "test.csv"
)

tfidf_path = os.path.join(
    PHASE2_DIR,
    "tfidf_vectorizer.pkl"
)


print("=" * 70)
print("PHASE 3 — BASELINE MODELS")
print("=" * 70)

print("\nLoading:")
print(train_path)
print(val_path)
print(test_path)
print(tfidf_path)


train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)
test_df = pd.read_csv(test_path)

tfidf = joblib.load(
    tfidf_path
)


print(
    f"\nTrain: {train_df.shape}"
)

print(
    f"Validation: {val_df.shape}"
)

print(
    f"Test: {test_df.shape}"
)


# ==============================================================
# 2. VERIFY REQUIRED FEATURES
# ==============================================================

extra_feats = [

    "has_url",
    "has_phone_or_code",
    "has_currency",
    "has_email",
    "has_upi_mention",
    "urgency_score",
    "exclamation_count",
    "digit_ratio",
    "special_char_ratio",
    "caps_ratio",
    "msg_length",
    "fraud_signal_count",

]


required_columns = [
    TEXT_COL,
    LABEL_COL,
    LANG_COL,
] + extra_feats


for dataset_name, dataset in [
    ("train", train_df),
    ("validation", val_df),
    ("test", test_df),
]:

    missing = [
        col
        for col in required_columns
        if col not in dataset.columns
    ]

    if missing:

        raise ValueError(
            f"{dataset_name} is missing columns: "
            f"{missing}"
        )


print(
    "\nAll required columns verified."
)


# ==============================================================
# 3. TF-IDF TRANSFORMATION
# ==============================================================

print("\n" + "=" * 70)
print("TF-IDF TRANSFORMATION")
print("=" * 70)


X_train_tfidf = tfidf.transform(
    train_df[TEXT_COL]
)

X_val_tfidf = tfidf.transform(
    val_df[TEXT_COL]
)

X_test_tfidf = tfidf.transform(
    test_df[TEXT_COL]
)


y_train = train_df[
    LABEL_COL
].values

y_val = val_df[
    LABEL_COL
].values

y_test = test_df[
    LABEL_COL
].values


print(
    f"\nTrain TF-IDF shape: "
    f"{X_train_tfidf.shape}"
)

print(
    f"Validation TF-IDF shape: "
    f"{X_val_tfidf.shape}"
)

print(
    f"Test TF-IDF shape: "
    f"{X_test_tfidf.shape}"
)


# ==============================================================
# 4. ADD NUMERIC FRAUD FEATURES
# ==============================================================

def add_extra_features(
    X,
    dataframe
):

    extra = csr_matrix(
        dataframe[
            extra_feats
        ]
        .astype(float)
        .values
    )

    return hstack(
        [X, extra],
        format="csr"
    )


X_train_full = add_extra_features(
    X_train_tfidf,
    train_df
)

X_val_full = add_extra_features(
    X_val_tfidf,
    val_df
)

X_test_full = add_extra_features(
    X_test_tfidf,
    test_df
)


print(
    f"\nTrain TF-IDF + fraud features: "
    f"{X_train_full.shape}"
)


# ==============================================================
# 5. DEFINE MODELS
# ==============================================================

models = {

    "logistic_regression":
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ),

    "linear_svm":
        LinearSVC(
            class_weight="balanced",
            random_state=42
        ),

    "naive_bayes":
        MultinomialNB()

}


# ==============================================================
# 6. VALIDATION EVALUATION
# ==============================================================

print("\n" + "=" * 70)
print("VALIDATION RESULTS")
print("=" * 70)


validation_results = {}


for name, model in models.items():

    print(
        f"\nTraining {name}..."
    )


    # Naive Bayes uses TF-IDF only because
    # it requires non-negative features.

    if name == "naive_bayes":

        Xtr = X_train_tfidf
        Xval = X_val_tfidf

    else:

        Xtr = X_train_full
        Xval = X_val_full


    model.fit(
        Xtr,
        y_train
    )


    val_preds = model.predict(
        Xval
    )


    accuracy = accuracy_score(
        y_val,
        val_preds
    )

    precision = precision_score(
        y_val,
        val_preds,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        val_preds,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        val_preds,
        zero_division=0
    )


    validation_results[name] = {

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "model": model,

        "preds": val_preds

    }


    print(
        f"{name.upper()}"
    )

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1       : {f1:.4f}"
    )


# ==============================================================
# 7. SELECT BEST MODEL USING VALIDATION F1
# ==============================================================

best_name = max(
    validation_results,
    key=lambda name:
    validation_results[name]["f1"]
)


print(
    "\n" + "=" * 70
)

print(
    f"BEST MODEL BASED ON VALIDATION F1: "
    f"{best_name}"
)

print(
    "=" * 70
)


# ==============================================================
# 8. FINAL TEST EVALUATION
# ==============================================================

print("\n" + "=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)


test_results = {}


for name in models:

    model = validation_results[
        name
    ]["model"]


    if name == "naive_bayes":

        Xtest = X_test_tfidf

    else:

        Xtest = X_test_full


    test_preds = model.predict(
        Xtest
    )


    accuracy = accuracy_score(
        y_test,
        test_preds
    )

    precision = precision_score(
        y_test,
        test_preds,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        test_preds,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        test_preds,
        zero_division=0
    )


    test_results[name] = {

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "preds": test_preds

    }


    print(
        f"\n{name.upper()}"
    )

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1       : {f1:.4f}"
    )


# ==============================================================
# 9. SAVE MODELS
# ==============================================================

print(
    "\nSaving trained models..."
)


for name, result in validation_results.items():

    model_path = os.path.join(
        MODEL_DIR,
        f"{name}.pkl"
    )

    joblib.dump(
        result["model"],
        model_path
    )

    print(
        f"Saved: {model_path}"
    )


# ==============================================================
# 10. BEST MODEL TEST REPORT
# ==============================================================

best_test_preds = test_results[
    best_name
]["preds"]


print(
    "\n" + "=" * 70
)

print(
    f"FINAL TEST REPORT — {best_name.upper()}"
)

print(
    "=" * 70
)

print(
    classification_report(
        y_test,
        best_test_preds,
        target_names=[
            "ham",
            "spam"
        ],
        zero_division=0
    )
)


# ==============================================================
# 11. CONFUSION MATRIX
# ==============================================================

cm = confusion_matrix(
    y_test,
    best_test_preds
)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "ham",
        "spam"
    ]
)


disp.plot(
    cmap="Blues"
)

plt.title(
    f"Confusion Matrix — {best_name}"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT_DIR,
        "baseline_confusion_matrix.png"
    ),
    dpi=150
)

plt.close()


# ==============================================================
# 12. MODEL COMPARISON
# ==============================================================

comparison_rows = []


for name in test_results:

    comparison_rows.append({

        "model": name,

        "accuracy":
            test_results[name]["accuracy"],

        "precision":
            test_results[name]["precision"],

        "recall":
            test_results[name]["recall"],

        "f1":
            test_results[name]["f1"]

    })


comparison_df = pd.DataFrame(
    comparison_rows
)


print(
    "\nModel comparison:"
)

print(
    comparison_df.to_string(
        index=False
    )
)


ax = comparison_df.set_index(
    "model"
)[
    [
        "precision",
        "recall",
        "f1"
    ]
].plot(
    kind="bar",
    figsize=(9, 5)
)


plt.title(
    "Baseline Model Comparison — Test Set"
)

plt.ylabel(
    "Score"
)

plt.ylim(
    0,
    1
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT_DIR,
        "model_comparison.png"
    ),
    dpi=150
)

plt.close()


# ==============================================================
# 13. PER-LANGUAGE PERFORMANCE
# ==============================================================

test_eval = test_df.copy()

test_eval["pred"] = best_test_preds


per_language_rows = []


for language, group in test_eval.groupby(
    LANG_COL
):

    language_y = group[
        LABEL_COL
    ]

    language_pred = group[
        "pred"
    ]


    per_language_rows.append({

        "language":
            language,

        "accuracy":
            accuracy_score(
                language_y,
                language_pred
            ),

        "precision":
            precision_score(
                language_y,
                language_pred,
                zero_division=0
            ),

        "recall":
            recall_score(
                language_y,
                language_pred,
                zero_division=0
            ),

        "f1":
            f1_score(
                language_y,
                language_pred,
                zero_division=0
            )

    })


per_language_df = pd.DataFrame(
    per_language_rows
)


print(
    "\nPer-language performance:"
)

print(
    per_language_df.to_string(
        index=False
    )
)


# ==============================================================
# 14. SAVE REPORT
# ==============================================================

report_path = os.path.join(
    BASE_DIR,
    "baseline_results.txt"
)


with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "PHASE 3 — BASELINE MODEL RESULTS\n"
    )

    f.write(
        "=" * 60 + "\n\n"
    )


    f.write(
        "IMPORTANT:\n"
    )

    f.write(
        "Models were selected using the validation set.\n"
    )

    f.write(
        "The test set was used only for final evaluation.\n\n"
    )


    f.write(
        "VALIDATION RESULTS\n"
    )

    f.write(
        "-" * 40 + "\n"
    )


    for name, result in validation_results.items():

        f.write(
            f"\n{name}\n"
        )

        f.write(
            f"Accuracy : "
            f"{result['accuracy']:.4f}\n"
        )

        f.write(
            f"Precision: "
            f"{result['precision']:.4f}\n"
        )

        f.write(
            f"Recall   : "
            f"{result['recall']:.4f}\n"
        )

        f.write(
            f"F1       : "
            f"{result['f1']:.4f}\n"
        )


    f.write(
        "\n\nFINAL TEST RESULTS\n"
    )

    f.write(
        "-" * 40 + "\n"
    )


    for name, result in test_results.items():

        f.write(
            f"\n{name}\n"
        )

        f.write(
            f"Accuracy : "
            f"{result['accuracy']:.4f}\n"
        )

        f.write(
            f"Precision: "
            f"{result['precision']:.4f}\n"
        )

        f.write(
            f"Recall   : "
            f"{result['recall']:.4f}\n"
        )

        f.write(
            f"F1       : "
            f"{result['f1']:.4f}\n"
        )


    f.write(
        f"\n\nBEST MODEL: {best_name}\n"
    )


    f.write(
        "\n\nBEST MODEL — CLASSIFICATION REPORT\n"
    )

    f.write(
        "-" * 40 + "\n"
    )

    f.write(
        classification_report(
            y_test,
            best_test_preds,
            target_names=[
                "ham",
                "spam"
            ],
            zero_division=0
        )
    )


    f.write(
        "\n\nPER-LANGUAGE PERFORMANCE\n"
    )

    f.write(
        "-" * 40 + "\n"
    )

    f.write(
        per_language_df.to_string(
            index=False
        )
    )


print(
    "\n" + "=" * 70
)

print(
    "PHASE 3 COMPLETE"
)

print(
    "=" * 70
)

print(
    f"\nBest baseline: {best_name}"
)

print(
    "\nGenerated:"
)

print(
    "  models/"
)

print(
    "  baseline_results.txt"
)

print(
    "  eda_plots/baseline_confusion_matrix.png"
)

print(
    "  eda_plots/model_comparison.png"
)

print(
    "\nNext:"
)

print(
    "PHASE 4 — MuRIL Transformer Fine-Tuning"
)

print(
    "=" * 70
)