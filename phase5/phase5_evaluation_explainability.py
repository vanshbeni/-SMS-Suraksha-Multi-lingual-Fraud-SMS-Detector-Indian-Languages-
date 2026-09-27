"""
PHASE 5 — Evaluation & Explainability
Project: SMS Suraksha — Multi-lingual Indian Fraud SMS Detector

This phase:

1. Loads the fine-tuned MuRIL model from Phase 4
2. Evaluates the model on the untouched Phase 2 test set
3. Generates:
      - Accuracy
      - Precision
      - Recall
      - F1
      - Classification report
      - Confusion matrix
      - Per-language performance
4. Generates LIME explanations for sample spam predictions
5. Saves all results for Phase 6

IMPORTANT:
    This script evaluates the FINAL 10K MuRIL model.

    Phase 4 model:
        ../phase4/muril_fraud_model_10k/

    Test data:
        ../phase2/test.csv

Run:
    pip install lime transformers torch scikit-learn matplotlib pandas
"""

# ==============================================================
# IMPORTS
# ==============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from lime.lime_text import LimeTextExplainer


# ==============================================================
# 0. PATH CONFIGURATION
# ==============================================================

# Current directory = phase5
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# Phase 2 contains test.csv
PHASE2_DIR = os.path.join(
    BASE_DIR,
    "..",
    "phase2"
)

# Phase 4 contains the trained 10K MuRIL model
PHASE4_MODEL_DIR = os.path.join(
    BASE_DIR,
    "..",
    "phase4",
    "muril_fraud_model_10k"
)

# Output directory
OUT_DIR = os.path.join(
    BASE_DIR,
    "eda_plots"
)

os.makedirs(
    OUT_DIR,
    exist_ok=True
)

# Report file
REPORT_FILE = os.path.join(
    BASE_DIR,
    "phase5_evaluation_report.txt"
)

# ==============================================================
# MODEL / DATA CONFIGURATION
# ==============================================================

MODEL_DIR = PHASE4_MODEL_DIR

TEST_FILE = os.path.join(
    PHASE2_DIR,
    "test.csv"
)

TEXT_COL = "text"
LABEL_COL = "label_binary"
LANG_COL = "language"

# IMPORTANT:
# Phase 4 used MAX_LEN = 64.
MAX_LEN = 64

# Batch size for full test-set prediction.
BATCH_SIZE = 64

# Number of LIME samples per explanation.
# 100 is faster than 200 while still providing useful explanations.
LIME_SAMPLES = 100

# Number of features shown in each LIME explanation.
LIME_FEATURES = 8


# ==============================================================
# 1. HEADER
# ==============================================================

print("=" * 70)
print("PHASE 5 — EVALUATION & EXPLAINABILITY")
print("=" * 70)

print("\nConfiguration:")
print(
    f"MuRIL model : {MODEL_DIR}"
)

print(
    f"Test data   : {TEST_FILE}"
)

print(
    f"MAX_LEN     : {MAX_LEN}"
)

print(
    f"Batch size  : {BATCH_SIZE}"
)

print(
    f"LIME samples: {LIME_SAMPLES}"
)


# ==============================================================
# 2. CHECK PATHS
# ==============================================================

print("\n" + "=" * 70)
print("CHECKING FILES")
print("=" * 70)


if not os.path.exists(MODEL_DIR):

    raise FileNotFoundError(
        "\nMuRIL model directory was not found:\n"
        f"{MODEL_DIR}\n\n"
        "Make sure Phase 4 created:\n"
        "phase4/muril_fraud_model_10k/"
    )


if not os.path.exists(TEST_FILE):

    raise FileNotFoundError(
        "\nTest dataset was not found:\n"
        f"{TEST_FILE}\n\n"
        "Make sure Phase 2 created:\n"
        "phase2/test.csv"
    )


print(
    "\nModel directory found."
)

print(
    "Test dataset found."
)


# ==============================================================
# 3. DEVICE
# ==============================================================

print("\n" + "=" * 70)
print("DEVICE")
print("=" * 70)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    "\nPyTorch version:"
)

print(
    torch.__version__
)


print(
    "\nCUDA available:"
)

print(
    torch.cuda.is_available()
)


print(
    "\nUsing device:"
)

print(
    device
)


if torch.cuda.is_available():

    print(
        "\nGPU:"
    )

    print(
        torch.cuda.get_device_name(0)
    )

    gpu_memory = round(
        torch.cuda.get_device_properties(
            0
        ).total_memory
        / (1024 ** 3),
        2
    )

    print(
        f"\nGPU VRAM: {gpu_memory} GB"
    )


# ==============================================================
# 4. LOAD TOKENIZER
# ==============================================================

print("\n" + "=" * 70)
print("LOADING MURIL TOKENIZER")
print("=" * 70)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR
)


print(
    "\nTokenizer loaded successfully."
)


# ==============================================================
# 5. LOAD MODEL
# ==============================================================

print("\n" + "=" * 70)
print("LOADING FINE-TUNED MURIL MODEL")
print("=" * 70)


model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
)


model = model.to(
    device
)


model.eval()


print(
    "\nMuRIL model loaded successfully."
)

print(
    "Model is in evaluation mode."
)


# ==============================================================
# 6. LOAD TEST DATA
# ==============================================================

print("\n" + "=" * 70)
print("LOADING TEST DATA")
print("=" * 70)


test_df = pd.read_csv(
    TEST_FILE
)


print(
    f"\nTest dataset shape: {test_df.shape}"
)


print(
    "\nColumns:"
)

print(
    test_df.columns.tolist()
)


# ==============================================================
# 7. VALIDATE REQUIRED COLUMNS
# ==============================================================

required_columns = [
    TEXT_COL,
    LABEL_COL,
    LANG_COL,
]


missing_columns = [
    column
    for column in required_columns
    if column not in test_df.columns
]


if missing_columns:

    raise ValueError(
        "Missing required columns: "
        f"{missing_columns}"
    )


print(
    "\nRequired columns verified."
)


# ==============================================================
# 8. TEST DATA SUMMARY
# ==============================================================

print("\n" + "=" * 70)
print("TEST DATA SUMMARY")
print("=" * 70)


print(
    "\nLabel distribution:"
)

print(
    test_df[
        LABEL_COL
    ].value_counts()
)


print(
    "\nLanguage distribution:"
)

print(
    test_df[
        LANG_COL
    ].value_counts()
)


# ==============================================================
# 9. PREDICTION FUNCTION
# ==============================================================

@torch.no_grad()
def predict_proba(
    texts
):
    """
    Return probabilities for each input message.

    Output shape:
        (n, 2)

    Column 0:
        P(ham)

    Column 1:
        P(spam)

    This format is required by LIME.
    """

    # Convert input to list.
    texts = list(texts)

    # Tokenize.
    encoded = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )

    # Move tensors to GPU/CPU.
    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    # Model prediction.
    outputs = model(
        **encoded
    )

    # Convert logits to probabilities.
    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )

    # Move back to CPU.
    probabilities = (
        probabilities
        .cpu()
        .numpy()
    )

    return probabilities


# ==============================================================
# 10. QUICK MODEL TEST
# ==============================================================

print("\n" + "=" * 70)
print("TESTING MODEL PREDICTION")
print("=" * 70)


sample_text = [
    "Congratulations! You have won a prize. Click the link now."
]


sample_probs = predict_proba(
    sample_text
)


print(
    "\nSample prediction:"
)

print(
    f"Ham probability : "
    f"{sample_probs[0][0]:.4f}"
)

print(
    f"Spam probability: "
    f"{sample_probs[0][1]:.4f}"
)


# ==============================================================
# 11. FULL TEST-SET PREDICTION
# ==============================================================

print("\n" + "=" * 70)
print("FULL TEST-SET PREDICTION")
print("=" * 70)


all_predictions = []
all_probabilities = []


total_rows = len(
    test_df
)


for start in range(
    0,
    total_rows,
    BATCH_SIZE
):

    end = min(
        start + BATCH_SIZE,
        total_rows
    )


    batch_texts = (
        test_df[
            TEXT_COL
        ]
        .iloc[start:end]
        .tolist()
    )


    probabilities = predict_proba(
        batch_texts
    )


    predictions = np.argmax(
        probabilities,
        axis=1
    )


    all_predictions.extend(
        predictions.tolist()
    )


    all_probabilities.extend(
        probabilities.tolist()
    )


    print(
        f"Processed "
        f"{end}/{total_rows}"
    )


# Add predictions to dataframe.
test_df["pred"] = (
    all_predictions
)


# Add probabilities.
test_df["ham_probability"] = [
    probability[0]
    for probability in all_probabilities
]


test_df["spam_probability"] = [
    probability[1]
    for probability in all_probabilities
]


print(
    "\nTest-set prediction complete."
)


# ==============================================================
# 12. OVERALL METRICS
# ==============================================================

print("\n" + "=" * 70)
print("OVERALL TEST RESULTS")
print("=" * 70)


y_true = test_df[
    LABEL_COL
].values


y_pred = test_df[
    "pred"
].values


accuracy = accuracy_score(
    y_true,
    y_pred
)


precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)


recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)


f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)


print(
    f"\nAccuracy : {accuracy:.4f}"
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
# 13. CLASSIFICATION REPORT
# ==============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)


report = classification_report(
    y_true,
    y_pred,
    target_names=[
        "ham",
        "spam"
    ],
    zero_division=0,
)


print(
    "\n"
)

print(
    report
)


# ==============================================================
# 14. CONFUSION MATRIX
# ==============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)


cm = confusion_matrix(
    y_true,
    y_pred
)


print(
    "\nConfusion matrix:"
)

print(
    cm
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
    "MuRIL — Test Set Confusion Matrix"
)


plt.tight_layout()


CONFUSION_PATH = os.path.join(
    OUT_DIR,
    "final_confusion_matrix.png"
)


plt.savefig(
    CONFUSION_PATH,
    dpi=150
)


plt.close()


print(
    "\nSaved:"
)

print(
    CONFUSION_PATH
)


# ==============================================================
# 15. PER-LANGUAGE EVALUATION
# ==============================================================

print("\n" + "=" * 70)
print("PER-LANGUAGE EVALUATION")
print("=" * 70)


per_language_rows = []


for language, group in test_df.groupby(
    LANG_COL
):

    language_true = group[
        LABEL_COL
    ].values


    language_pred = group[
        "pred"
    ].values


    language_accuracy = accuracy_score(
        language_true,
        language_pred
    )


    language_precision = precision_score(
        language_true,
        language_pred,
        zero_division=0
    )


    language_recall = recall_score(
        language_true,
        language_pred,
        zero_division=0
    )


    language_f1 = f1_score(
        language_true,
        language_pred,
        zero_division=0
    )


    per_language_rows.append({

        "language":
            language,

        "n":
            len(group),

        "accuracy":
            language_accuracy,

        "precision":
            language_precision,

        "recall":
            language_recall,

        "f1":
            language_f1,
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
# 16. PER-LANGUAGE ACCURACY PLOT
# ==============================================================

plt.figure(
    figsize=(8, 5)
)


per_language_df.set_index(
    "language"
)["accuracy"].plot(
    kind="bar"
)


plt.title(
    "MuRIL Accuracy by Language"
)

plt.ylabel(
    "Accuracy"
)

plt.xlabel(
    "Language"
)

plt.ylim(
    0,
    1
)


plt.tight_layout()


LANGUAGE_PLOT_PATH = os.path.join(
    OUT_DIR,
    "per_language_accuracy.png"
)


plt.savefig(
    LANGUAGE_PLOT_PATH,
    dpi=150
)


plt.close()


print(
    "\nSaved:"
)

print(
    LANGUAGE_PLOT_PATH
)


# ==============================================================
# 17. LIME EXPLAINABILITY
# ==============================================================

print("\n" + "=" * 70)
print("LIME EXPLAINABILITY")
print("=" * 70)


explainer = LimeTextExplainer(
    class_names=[
        "ham",
        "spam"
    ]
)


def explain_message(
    text,
    num_features=LIME_FEATURES
):
    """
    Generate a LIME explanation.

    Positive weights:
        Push prediction toward spam.

    Negative weights:
        Push prediction toward ham.
    """

    explanation = (
        explainer.explain_instance(
            text,
            predict_proba,
            num_features=num_features,
            num_samples=LIME_SAMPLES,
        )
    )

    return explanation.as_list()


# ==============================================================
# 18. SELECT SAMPLE SPAM PREDICTIONS
# ==============================================================

spam_predictions = test_df[
    test_df["pred"] == 1
]


print(
    f"\nSpam predictions available: "
    f"{len(spam_predictions)}"
)


sample_count = min(
    5,
    len(spam_predictions)
)


if sample_count > 0:

    sample_spam = (
        spam_predictions
        .sample(
            sample_count,
            random_state=42
        )
    )

else:

    sample_spam = (
        test_df
        .sample(
            min(5, len(test_df)),
            random_state=42
        )
    )


# ==============================================================
# 19. GENERATE LIME EXPLANATIONS
# ==============================================================

explanations_log = []


for index, row in sample_spam.iterrows():

    text = row[
        TEXT_COL
    ]


    language = row[
        LANG_COL
    ]


    prediction = int(
        row["pred"]
    )


    spam_probability = float(
        row["spam_probability"]
    )


    print(
        "\n" + "-" * 70
    )


    print(
        f"Message ({language}):"
    )


    print(
        text
    )


    print(
        f"\nPredicted class: "
        f"{'SPAM' if prediction == 1 else 'HAM'}"
    )


    print(
        f"Spam probability: "
        f"{spam_probability:.4f}"
    )


    print(
        "\nGenerating LIME explanation..."
    )


    explanation = explain_message(
        text
    )


    explanations_log.append({

        "text":
            text,

        "language":
            language,

        "prediction":
            prediction,

        "spam_probability":
            spam_probability,

        "top_contributing_words":
            explanation,
    })


    print(
        "\nTop contributing tokens:"
    )


    for token, weight in explanation:

        direction = (
            "SPAM"
            if weight > 0
            else "HAM"
        )


        print(
            f"  {token!r:30s}"
            f" {weight:+.4f}"
            f"  -> {direction}"
        )


# ==============================================================
# 20. SAVE LIME EXPLANATIONS
# ==============================================================

LIME_FILE = os.path.join(
    BASE_DIR,
    "lime_explanations.txt"
)


with open(
    LIME_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "PHASE 5 — LIME EXPLANATIONS\n"
    )

    f.write(
        "=" * 60
        + "\n\n"
    )


    for item in explanations_log:

        f.write(
            f"Language: "
            f"{item['language']}\n"
        )


        f.write(
            f"Prediction: "
            f"{'SPAM' if item['prediction'] == 1 else 'HAM'}\n"
        )


        f.write(
            f"Spam probability: "
            f"{item['spam_probability']:.4f}\n"
        )


        f.write(
            f"Message:\n"
            f"{item['text']}\n\n"
        )


        f.write(
            "Top contributing tokens:\n"
        )


        for token, weight in (
            item[
                "top_contributing_words"
            ]
        ):

            direction = (
                "SPAM"
                if weight > 0
                else "HAM"
            )


            f.write(
                f"  {token} : "
                f"{weight:+.4f} "
                f"({direction})\n"
            )


        f.write(
            "\n"
            + "-" * 60
            + "\n\n"
        )


print(
    "\nLIME explanations saved:"
)

print(
    LIME_FILE
)


# ==============================================================
# 21. SAVE COMPLETE TEST PREDICTIONS
# ==============================================================

PREDICTIONS_FILE = os.path.join(
    BASE_DIR,
    "phase5_test_predictions.csv"
)


test_df.to_csv(
    PREDICTIONS_FILE,
    index=False
)


print(
    "\nTest predictions saved:"
)

print(
    PREDICTIONS_FILE
)


# ==============================================================
# 22. SAVE FINAL REPORT
# ==============================================================

print("\n" + "=" * 70)
print("SAVING FINAL REPORT")
print("=" * 70)


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "PHASE 5 — FINAL EVALUATION & EXPLAINABILITY\n"
    )

    f.write(
        "=" * 70
        + "\n\n"
    )


    f.write(
        "MODEL\n"
    )

    f.write(
        "-" * 40
        + "\n"
    )

    f.write(
        "Fine-tuned model: MuRIL\n"
    )

    f.write(
        "Model directory: "
        f"{MODEL_DIR}\n"
    )

    f.write(
        f"Maximum sequence length: "
        f"{MAX_LEN}\n\n"
    )


    f.write(
        "TEST DATA\n"
    )

    f.write(
        "-" * 40
        + "\n"
    )

    f.write(
        f"Test rows: "
        f"{len(test_df)}\n"
    )

    f.write(
        f"Languages: "
        f"{sorted(test_df[LANG_COL].unique())}\n\n"
    )


    f.write(
        "OVERALL TEST METRICS\n"
    )

    f.write(
        "-" * 40
        + "\n"
    )

    f.write(
        f"Accuracy : {accuracy:.4f}\n"
    )

    f.write(
        f"Precision: {precision:.4f}\n"
    )

    f.write(
        f"Recall   : {recall:.4f}\n"
    )

    f.write(
        f"F1       : {f1:.4f}\n\n"
    )


    f.write(
        "CLASSIFICATION REPORT\n"
    )

    f.write(
        "-" * 40
        + "\n"
    )

    f.write(
        report
    )


    f.write(
        "\n\n"
    )


    f.write(
        "CONFUSION MATRIX\n"
    )

    f.write(
        "-" * 40
        + "\n"
    )

    f.write(
        str(cm)
    )


    f.write(
        "\n\n"
    )


    f.write(
        "PER-LANGUAGE PERFORMANCE\n"
    )

    f.write(
        "-" * 40
        + "\n"
    )

    f.write(
        per_language_df.to_string(
            index=False
        )
    )


    f.write(
        "\n\n"
    )


    f.write(
        "LIME EXPLANATIONS\n"
    )

    f.write(
        "-" * 40
        + "\n"
    )


    for item in explanations_log:

        f.write(
            f"\n[{item['language']}] "
            f"{item['text']}\n"
        )


        f.write(
            f"Prediction: "
            f"{'SPAM' if item['prediction'] == 1 else 'HAM'}\n"
        )


        f.write(
            f"Spam probability: "
            f"{item['spam_probability']:.4f}\n"
        )


        f.write(
            "Contributing tokens:\n"
        )


        f.write(
            str(
                item[
                    "top_contributing_words"
                ]
            )
        )


        f.write(
            "\n"
        )


print(
    "\nReport saved:"
)

print(
    REPORT_FILE
)


# ==============================================================
# 23. FINAL SUMMARY
# ==============================================================

print("\n" + "=" * 70)
print("PHASE 5 COMPLETE")
print("=" * 70)


print(
    "\nFINAL MURIL TEST RESULTS"
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


print(
    "\nLinear SVM baseline:"
)

print(
    "Accuracy : 0.9826"
)

print(
    "Precision: 0.9371"
)

print(
    "Recall   : 0.9221"
)

print(
    "F1       : 0.9295"
)


print(
    "\nMuRIL:"
)

print(
    f"F1       : {f1:.4f}"
)


if f1 > 0.9295:

    print(
        "\nMuRIL beats the Linear SVM baseline."
    )

elif f1 == 0.9295:

    print(
        "\nMuRIL matches the Linear SVM baseline."
    )

else:

    print(
        "\nLinear SVM remains the best overall model."
    )


print(
    "\nGenerated files:"
)

print(
    "  phase5_evaluation_report.txt"
)

print(
    "  lime_explanations.txt"
)

print(
    "  phase5_test_predictions.csv"
)

print(
    "  eda_plots/final_confusion_matrix.png"
)

print(
    "  eda_plots/per_language_accuracy.png"
)


print(
    "\nReady for Phase 6 — API + User Interface."
)

print(
    "=" * 70
)