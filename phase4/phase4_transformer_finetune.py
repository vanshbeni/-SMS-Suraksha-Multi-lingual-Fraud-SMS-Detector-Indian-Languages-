"""
PHASE 4 — MuRIL Transformer Fine-tuning
SECOND / STRONGER EXPERIMENT

Project:
    SMS Suraksha — Multi-lingual Indian Fraud SMS Detector

Model:
    google/muril-base-cased

Languages:
    English, Hindi, Bengali, Urdu, Punjabi, Marathi

INPUT:
    ../phase2/train.csv
    ../phase2/val.csv
    ../phase2/test.csv

TRAINING CONFIGURATION:
    Training rows: approximately 10,000
    Epochs: 3
    Maximum sequence length: 64
    GPU batch size: 4
    Gradient accumulation: 1
    FP16: enabled

IMPORTANT:
    Only the TRAINING set is reduced.
    Validation and TEST sets remain completely untouched.

    Training rows are selected by message_id groups so that
    translations of the same original SMS remain together.

OUTPUT:
    muril_fraud_model_10k/
    transformer_results_10k.txt

BASELINE:
    Linear SVM test F1 = 0.9295

GOAL:
    Determine whether MuRIL trained on substantially more data
    can approach or beat the Linear SVM baseline.
"""

# ==============================================================
# IMPORTS
# ==============================================================

import os
import numpy as np
import pandas as pd
import torch

from datasets import Dataset

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)


# ==============================================================
# 0. CONFIGURATION
# ==============================================================

MODEL_NAME = "google/muril-base-cased"

TEXT_COL = "text"
LABEL_COL = "label_binary"
LANG_COL = "language"
MESSAGE_ID_COL = "message_id"

# --------------------------------------------------------------
# EXPERIMENT SETTINGS
# --------------------------------------------------------------

FAST_MODE = True

# Target number of training rows.
# Complete message_id groups are selected.
MAX_TRAIN_ROWS = 10000

# SMS messages are short.
MAX_LEN = 64

RANDOM_SEED = 42

NUM_EPOCHS = 3

# ==============================================================
# PATHS
# ==============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PHASE2_DIR = os.path.join(
    BASE_DIR,
    "..",
    "phase2"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "muril_fraud_model_10k"
)

CHECKPOINT_DIR = os.path.join(
    BASE_DIR,
    "muril_checkpoints_10k"
)

RESULTS_FILE = os.path.join(
    BASE_DIR,
    "transformer_results_10k.txt"
)


# ==============================================================
# 1. HEADER
# ==============================================================

print("=" * 70)
print("PHASE 4 — MURIL SECOND EXPERIMENT")
print("=" * 70)

print("\nExperiment configuration:")
print(f"Model: {MODEL_NAME}")
print(f"Training rows target: {MAX_TRAIN_ROWS}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Maximum sequence length: {MAX_LEN}")
print(f"Random seed: {RANDOM_SEED}")


# ==============================================================
# 2. GPU CHECK
# ==============================================================

print("\n" + "=" * 70)
print("GPU CHECK")
print("=" * 70)

print("\nPyTorch:")
print(torch.__version__)

print("\nCUDA available:")
print(torch.cuda.is_available())


if not torch.cuda.is_available():

    raise RuntimeError(
        "\nCUDA is not available.\n"
        "This experiment is designed for your NVIDIA RTX 3050 GPU.\n"
        "Please fix CUDA before continuing."
    )


GPU_NAME = torch.cuda.get_device_name(0)

GPU_MEMORY = round(
    torch.cuda.get_device_properties(0).total_memory
    / (1024 ** 3),
    2
)

print("\nGPU:")
print(GPU_NAME)

print("\nGPU memory:")
print(f"{GPU_MEMORY} GB")

print("\nPyTorch CUDA:")
print(torch.version.cuda)


# ==============================================================
# 3. RANDOM SEEDS
# ==============================================================

np.random.seed(RANDOM_SEED)

torch.manual_seed(
    RANDOM_SEED
)

torch.cuda.manual_seed_all(
    RANDOM_SEED
)


# ==============================================================
# 4. LOAD DATA
# ==============================================================

print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)


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


print("\nTrain:")
print(train_path)

print("\nValidation:")
print(val_path)

print("\nTest:")
print(test_path)


train_full = pd.read_csv(
    train_path
)

val_full = pd.read_csv(
    val_path
)

test_full = pd.read_csv(
    test_path
)


print(
    f"\nOriginal train shape: {train_full.shape}"
)

print(
    f"Validation shape: {val_full.shape}"
)

print(
    f"Test shape: {test_full.shape}"
)


# ==============================================================
# 5. REQUIRED COLUMN CHECK
# ==============================================================

required_columns = [
    TEXT_COL,
    LABEL_COL,
    LANG_COL,
    MESSAGE_ID_COL,
]


for dataset_name, dataframe in [
    ("train", train_full),
    ("validation", val_full),
    ("test", test_full),
]:

    missing = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing:

        raise ValueError(
            f"{dataset_name} is missing: {missing}"
        )


print(
    "\nRequired columns verified."
)


# ==============================================================
# 6. TRANSLATION LEAKAGE CHECK
# ==============================================================

print("\n" + "=" * 70)
print("TRANSLATION LEAKAGE CHECK")
print("=" * 70)


train_ids = set(
    train_full[MESSAGE_ID_COL]
)

val_ids = set(
    val_full[MESSAGE_ID_COL]
)

test_ids = set(
    test_full[MESSAGE_ID_COL]
)


if not train_ids.isdisjoint(
    val_ids
):

    raise ValueError(
        "Translation leakage detected between "
        "train and validation."
    )


if not train_ids.isdisjoint(
    test_ids
):

    raise ValueError(
        "Translation leakage detected between "
        "train and test."
    )


if not val_ids.isdisjoint(
    test_ids
):

    raise ValueError(
        "Translation leakage detected between "
        "validation and test."
    )


print(
    "\nTranslation leakage check: PASSED"
)

print(
    f"Original train messages: {len(train_ids)}"
)

print(
    f"Original validation messages: {len(val_ids)}"
)

print(
    f"Original test messages: {len(test_ids)}"
)


# ==============================================================
# 7. CREATE 10,000-ROW TRAINING SUBSET
# ==============================================================

print("\n" + "=" * 70)
print("CREATING 10,000-ROW TRAINING SUBSET")
print("=" * 70)


original_train_rows = len(
    train_full
)


if (
    FAST_MODE
    and original_train_rows > MAX_TRAIN_ROWS
):

    rng = np.random.RandomState(
        RANDOM_SEED
    )

    # ----------------------------------------------------------
    # Group by original SMS.
    #
    # This keeps all available translations of an original
    # message together.
    # ----------------------------------------------------------

    grouped = list(
        train_full.groupby(
            MESSAGE_ID_COL,
            sort=False
        )
    )

    rng.shuffle(
        grouped
    )

    selected_message_ids = []

    selected_rows = 0


    for message_id, group in grouped:

        group_size = len(group)


        if (
            selected_rows + group_size
            > MAX_TRAIN_ROWS
        ):

            continue


        selected_message_ids.append(
            message_id
        )

        selected_rows += group_size


        if selected_rows >= MAX_TRAIN_ROWS:

            break


    train_full = train_full[
        train_full[
            MESSAGE_ID_COL
        ].isin(
            selected_message_ids
        )
    ].copy()


print(
    f"\nFinal training rows: "
    f"{len(train_full)}"
)

print(
    f"Original SMS groups: "
    f"{train_full[MESSAGE_ID_COL].nunique()}"
)


print(
    "\nTraining language distribution:"
)

print(
    train_full[
        LANG_COL
    ].value_counts()
)


print(
    "\nTraining label distribution:"
)

print(
    train_full[
        LABEL_COL
    ].value_counts()
)


# ==============================================================
# 8. VALIDATION / TEST ARE UNCHANGED
# ==============================================================

print(
    "\nValidation rows:"
)

print(
    len(val_full)
)

print(
    "Test rows:"
)

print(
    len(test_full)
)


# ==============================================================
# 9. CREATE HUGGING FACE DATASETS
# ==============================================================

train_df = train_full[
    [
        TEXT_COL,
        LABEL_COL
    ]
].rename(
    columns={
        LABEL_COL: "labels"
    }
)


val_df = val_full[
    [
        TEXT_COL,
        LABEL_COL
    ]
].rename(
    columns={
        LABEL_COL: "labels"
    }
)


test_df = test_full[
    [
        TEXT_COL,
        LABEL_COL
    ]
].rename(
    columns={
        LABEL_COL: "labels"
    }
)


train_ds = Dataset.from_pandas(
    train_df,
    preserve_index=False
)

val_ds = Dataset.from_pandas(
    val_df,
    preserve_index=False
)

test_ds = Dataset.from_pandas(
    test_df,
    preserve_index=False
)


# ==============================================================
# 10. TOKENIZER
# ==============================================================

print("\n" + "=" * 70)
print("LOADING MURIL TOKENIZER")
print("=" * 70)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


print(
    "\nTokenizer loaded."
)


# ==============================================================
# 11. TOKENIZATION
# ==============================================================

def tokenize_fn(batch):

    return tokenizer(
        batch[TEXT_COL],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
    )


print(
    "\nTokenizing train..."
)

train_ds = train_ds.map(
    tokenize_fn,
    batched=True
)


print(
    "Tokenizing validation..."
)

val_ds = val_ds.map(
    tokenize_fn,
    batched=True
)


print(
    "Tokenizing test..."
)

test_ds = test_ds.map(
    tokenize_fn,
    batched=True
)


print(
    "\nTokenization complete."
)


# ==============================================================
# 12. FORMAT DATASETS
# ==============================================================

model_columns = [
    "input_ids",
    "attention_mask",
    "labels",
]


if "token_type_ids" in train_ds.column_names:

    model_columns.insert(
        2,
        "token_type_ids"
    )


train_ds.set_format(
    type="torch",
    columns=model_columns
)

val_ds.set_format(
    type="torch",
    columns=model_columns
)

test_ds.set_format(
    type="torch",
    columns=model_columns
)


print(
    "\nModel input columns:"
)

print(
    model_columns
)


# ==============================================================
# 13. LOAD MURIL
# ==============================================================

print("\n" + "=" * 70)
print("LOADING MURIL MODEL")
print("=" * 70)


model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)


print(
    "\nMuRIL model loaded."
)


# ==============================================================
# 14. CLASS WEIGHTS
# ==============================================================

print("\n" + "=" * 70)
print("CLASS WEIGHTS")
print("=" * 70)


class_counts = (
    train_df[
        "labels"
    ]
    .value_counts()
    .to_dict()
)


total = sum(
    class_counts.values()
)


class_weights = torch.tensor(
    [
        total / (
            2 * class_counts.get(
                0,
                1
            )
        ),

        total / (
            2 * class_counts.get(
                1,
                1
            )
        ),
    ],
    dtype=torch.float,
)


print(
    "\nClass counts:"
)

print(
    class_counts
)


print(
    "\nClass weights:"
)

print(
    class_weights
)


# ==============================================================
# 15. METRICS
# ==============================================================

def compute_metrics(
    eval_pred
):

    logits = eval_pred.predictions

    labels = eval_pred.label_ids


    predictions = np.argmax(
        logits,
        axis=1
    )


    return {

        "accuracy":
            accuracy_score(
                labels,
                predictions
            ),

        "precision":
            precision_score(
                labels,
                predictions,
                zero_division=0
            ),

        "recall":
            recall_score(
                labels,
                predictions,
                zero_division=0
            ),

        "f1":
            f1_score(
                labels,
                predictions,
                zero_division=0
            ),
    }


# ==============================================================
# 16. WEIGHTED TRAINER
# ==============================================================

class WeightedTrainer(
    Trainer
):

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        **kwargs
    ):

        labels = inputs.pop(
            "labels"
        )


        outputs = model(
            **inputs
        )


        logits = outputs.logits


        loss_function = (
            torch.nn.CrossEntropyLoss(
                weight=class_weights.to(
                    logits.device
                )
            )
        )


        loss = loss_function(
            logits,
            labels
        )


        if return_outputs:

            return (
                loss,
                outputs
            )


        return loss


# ==============================================================
# 17. TRAINING ARGUMENTS
# ==============================================================

print("\n" + "=" * 70)
print("TRAINING CONFIGURATION")
print("=" * 70)


training_args = TrainingArguments(

    output_dir=CHECKPOINT_DIR,

    eval_strategy="epoch",

    save_strategy="epoch",

    learning_rate=2e-5,

    # RTX 3050 4 GB
    per_device_train_batch_size=4,

    per_device_eval_batch_size=8,

    # No accumulation for faster training.
    gradient_accumulation_steps=1,

    num_train_epochs=NUM_EPOCHS,

    weight_decay=0.01,

    load_best_model_at_end=True,

    metric_for_best_model="f1",

    greater_is_better=True,

    logging_steps=25,

    report_to=[],

    save_total_limit=2,

    fp16=True,

)


print(
    "\nTrain batch size:",
    training_args.per_device_train_batch_size
)

print(
    "Evaluation batch size:",
    training_args.per_device_eval_batch_size
)

print(
    "Gradient accumulation:",
    training_args.gradient_accumulation_steps
)

print(
    "Effective batch size:",
    training_args.per_device_train_batch_size
    * training_args.gradient_accumulation_steps
)

print(
    "Epochs:",
    NUM_EPOCHS
)

print(
    "MAX_LEN:",
    MAX_LEN
)

print(
    "FP16:",
    training_args.fp16
)


# ==============================================================
# 18. TRAINER
# ==============================================================

trainer = WeightedTrainer(

    model=model,

    args=training_args,

    train_dataset=train_ds,

    eval_dataset=val_ds,

    compute_metrics=compute_metrics,

    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=1
        )
    ],
)


# ==============================================================
# 19. TRAIN
# ==============================================================

print("\n" + "=" * 70)
print("STARTING MURIL 10K TRAINING")
print("=" * 70)


print(
    f"\nTraining rows: {len(train_ds)}"
)

print(
    f"Validation rows: {len(val_ds)}"
)

print(
    f"Test rows: {len(test_ds)}"
)

print(
    f"Epochs: {NUM_EPOCHS}"
)

print(
    f"MAX_LEN: {MAX_LEN}"
)

print(
    "\nValidation and test sets are NOT reduced."
)


trainer.train()


# ==============================================================
# 20. FINAL TEST EVALUATION
# ==============================================================

print("\n" + "=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)


test_output = trainer.predict(
    test_ds
)


test_predictions = np.argmax(
    test_output.predictions,
    axis=1
)


test_labels = test_df[
    "labels"
].values


accuracy = accuracy_score(
    test_labels,
    test_predictions
)


precision = precision_score(
    test_labels,
    test_predictions,
    zero_division=0
)


recall = recall_score(
    test_labels,
    test_predictions,
    zero_division=0
)


f1 = f1_score(
    test_labels,
    test_predictions,
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
# 21. CLASSIFICATION REPORT
# ==============================================================

report = classification_report(

    test_labels,

    test_predictions,

    target_names=[
        "ham",
        "spam"
    ],

    zero_division=0
)


print(
    "\nClassification Report:"
)

print(
    report
)


# ==============================================================
# 22. PER-LANGUAGE EVALUATION
# ==============================================================

print("\n" + "=" * 70)
print("PER-LANGUAGE PERFORMANCE")
print("=" * 70)


test_eval = test_full.copy()

test_eval[
    "pred"
] = test_predictions


per_language_rows = []


for language, group in test_eval.groupby(
    LANG_COL
):

    y_true = group[
        LABEL_COL
    ]

    y_pred = group[
        "pred"
    ]


    per_language_rows.append({

        "language":
            language,

        "accuracy":
            accuracy_score(
                y_true,
                y_pred
            ),

        "precision":
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "recall":
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "f1":
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            ),
    })


per_language_df = pd.DataFrame(
    per_language_rows
)


print(
    "\nPer-language results:"
)

print(
    per_language_df.to_string(
        index=False
    )
)


# ==============================================================
# 23. SAVE MODEL
# ==============================================================

print("\n" + "=" * 70)
print("SAVING MODEL")
print("=" * 70)


trainer.save_model(
    OUTPUT_DIR
)


tokenizer.save_pretrained(
    OUTPUT_DIR
)


print(
    "\nModel saved:"
)

print(
    OUTPUT_DIR
)


# ==============================================================
# 24. SAVE RESULTS
# ==============================================================

print(
    "\nSaving results..."
)


with open(
    RESULTS_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "PHASE 4 — MURIL 10K EXPERIMENT RESULTS\n"
    )

    f.write(
        "=" * 60 + "\n\n"
    )


    f.write(
        f"Model: {MODEL_NAME}\n"
    )

    f.write(
        f"Training rows: {len(train_full)}\n"
    )

    f.write(
        f"Training original SMS groups: "
        f"{train_full[MESSAGE_ID_COL].nunique()}\n"
    )

    f.write(
        f"Validation rows: {len(val_full)}\n"
    )

    f.write(
        f"Test rows: {len(test_full)}\n"
    )

    f.write(
        f"Epochs: {NUM_EPOCHS}\n"
    )

    f.write(
        f"MAX_LEN: {MAX_LEN}\n"
    )

    f.write(
        f"Batch size: "
        f"{training_args.per_device_train_batch_size}\n"
    )

    f.write(
        "\nFINAL TEST METRICS\n"
    )

    f.write(
        "-" * 40 + "\n"
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
        "-" * 40 + "\n"
    )

    f.write(
        report
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


# ==============================================================
# 25. BASELINE COMPARISON
# ==============================================================

print("\n" + "=" * 70)
print("BASELINE COMPARISON")
print("=" * 70)


SVM_F1 = 0.9295


print(
    f"\nLinear SVM F1 : {SVM_F1:.4f}"
)

print(
    f"MuRIL F1      : {f1:.4f}"
)


if f1 > SVM_F1:

    print(
        "\nMuRIL BEAT the Linear SVM baseline."
    )

elif f1 == SVM_F1:

    print(
        "\nMuRIL MATCHED the Linear SVM baseline."
    )

else:

    difference = SVM_F1 - f1

    print(
        "\nMuRIL is below the Linear SVM baseline."
    )

    print(
        f"F1 difference: {difference:.4f}"
    )


# ==============================================================
# 26. COMPLETE
# ==============================================================

print("\n" + "=" * 70)
print("PHASE 4 — 10K EXPERIMENT COMPLETE")
print("=" * 70)


print(
    "\nGenerated:"
)

print(
    "  muril_fraud_model_10k/"
)

print(
    "  transformer_results_10k.txt"
)


print(
    "\nNext decision:"
)

print(
    "Compare MuRIL against Linear SVM."
)

print(
    "Linear SVM baseline F1 = 0.9295"
)

print(
    f"MuRIL 10K F1 = {f1:.4f}"
)

print(
    "\nIf MuRIL is still below 0.9295, "
    "Linear SVM remains the best model."
)

print(
    "=" * 70
)