"""
PHASE 1 (CORRECTED) — Data Loading, Melting, Cleaning & EDA
Project: Multi-lingual (Indian) Fraud SMS Detector
Dataset: dbarbedillo/SMS_Spam_Multilingual_Collection_Dataset (Hugging Face)

ACTUAL SCHEMA (confirmed by running the loader):
  - one row per original message
  - label column: 'labels' (values: 'ham' / 'spam')
  - one text column PER LANGUAGE: 'text' (English), 'text_hi', 'text_de',
    'text_fr', 'text_es', 'text_zh', 'text_ar', 'text_bn', 'text_ru',
    'text_pt', 'text_ja', 'text_id', 'text_ur', 'text_pa', 'text_jv',
    'text_tr', 'text_ko', 'text_mr', 'text_uk', 'text_sv', 'text_no'

This script MELTS the wide table into long format — one row per
(message, language) pair — keeping only the Indian-language columns,
so each original message contributes up to 6 rows (en/hi/bn/ur/pa/mr),
all sharing the same ham/spam label.

Run: pip install datasets pandas matplotlib

Output:
  - cleaned_sms_dataset.csv   (long format: text, language, label)
  - eda_plots/*.png
  - phase1_report.txt
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = "eda_plots"
os.makedirs(OUT_DIR, exist_ok=True)

RAW_LABEL_COL = "labels"
TEXT_COL = "text"
LABEL_COL = "label"
LANG_COL = "language"

# Maps each Indian-language text column to its language code.
# Add/remove entries here if you want a different language scope.
INDIAN_LANG_COLUMNS = {
    "text": "en",
    "text_hi": "hi",
    "text_bn": "bn",
    "text_ur": "ur",
    "text_pa": "pa",
    "text_mr": "mr",
}

# ----------------------------------------------------------------------
# 1. LOAD DATASET
# ----------------------------------------------------------------------
def load_dataset_hf():
    try:
        from datasets import load_dataset
        ds = load_dataset("dbarbedillo/SMS_Spam_Multilingual_Collection_Dataset")
        df = ds["train"].to_pandas()
        print(f"Loaded from Hugging Face: {df.shape}")
        return df
    except Exception as e:
        print(f"HF load failed ({e}). Trying local file 'raw_dataset.csv' instead...")
        return pd.read_csv("raw_dataset.csv")


raw_df = load_dataset_hf()
print("Columns:", raw_df.columns.tolist())
raw_df["message_id"] = raw_df.index

missing = [c for c in INDIAN_LANG_COLUMNS if c not in raw_df.columns]
if missing:
    raise ValueError(
        f"Expected columns not found in dataset: {missing}. "
        f"Available columns: {raw_df.columns.tolist()}"
    )

# ----------------------------------------------------------------------
# 2. MELT WIDE -> LONG (one row per message-language pair)
# ----------------------------------------------------------------------
long_frames = []
for col, lang in INDIAN_LANG_COLUMNS.items():
    tmp = raw_df[["message_id", RAW_LABEL_COL, col]].rename(
        columns={
            col: TEXT_COL,
            RAW_LABEL_COL: LABEL_COL
        }
    )
    tmp[LANG_COL] = lang
    long_frames.append(tmp)

df = pd.concat(long_frames, ignore_index=True)
print(f"Melted to long format: {df.shape}")
print(df[LANG_COL].value_counts())

# ----------------------------------------------------------------------
# 3. BASIC CLEANING
# ----------------------------------------------------------------------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\u200b", "")
    return text


df[TEXT_COL] = df[TEXT_COL].apply(clean_text)
df = df[df[TEXT_COL].str.len() > 0]

# Dedup within each language separately (a short phrase like "Ok" can
# legitimately repeat across languages, so we don't dedup across them)
df = df.drop_duplicates(subset=[TEXT_COL, LANG_COL])
print(f"After cleaning + dedup: {df.shape}")
print(df[LANG_COL].value_counts())

# ----------------------------------------------------------------------
# 4. EDA PLOTS
# ----------------------------------------------------------------------
def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, name), dpi=150)
    plt.close()

plt.figure(figsize=(5, 4))
df[LABEL_COL].value_counts().plot(kind="bar", color=["#4CAF50", "#E53935"])
plt.title("Ham vs Spam distribution (Indian languages)")
savefig("class_balance.png")

plt.figure(figsize=(7, 4))
df[LANG_COL].value_counts().plot(kind="bar", color="#3F51B5")
plt.title("Messages per Indian language")
savefig("language_distribution.png")

spam_ratio = df.groupby(LANG_COL)[LABEL_COL].apply(lambda x: (x == "spam").mean()).sort_values(ascending=False)
plt.figure(figsize=(7, 4))
spam_ratio.plot(kind="bar", color="#FF9800")
plt.title("Spam ratio by Indian language")
savefig("spam_ratio_by_language.png")

df["msg_length"] = df[TEXT_COL].apply(len)
plt.figure(figsize=(7, 5))
for label, color in [("ham", "#4CAF50"), ("spam", "#E53935")]:
    subset = df[df[LABEL_COL] == label]["msg_length"]
    plt.hist(subset, bins=40, alpha=0.6, label=label, color=color)
plt.legend()
plt.title("Message length: ham vs spam")
savefig("message_length_dist.png")

# ----------------------------------------------------------------------
# 5. SAVE OUTPUTS
# ----------------------------------------------------------------------
df.to_csv("cleaned_sms_dataset.csv", index=False)

with open("phase1_report.txt", "w") as f:
    f.write("PHASE 1 SUMMARY — INDIAN LANGUAGES ONLY (melted long format)\n")
    f.write("=" * 55 + "\n")
    f.write(f"Languages included: {sorted(df[LANG_COL].unique())}\n")
    f.write(f"Total (message, language) rows: {len(df)}\n")
    f.write(f"Class balance:\n{df[LABEL_COL].value_counts().to_string()}\n\n")
    f.write(f"Rows per language:\n{df[LANG_COL].value_counts().to_string()}\n\n")
    f.write(f"Spam ratio by language:\n{spam_ratio.to_string()}\n")

print("\nPhase 1 complete. Files saved: cleaned_sms_dataset.csv, phase1_report.txt, eda_plots/")
print(f"\nFinal shape: {df.shape} across {df[LANG_COL].nunique()} languages")