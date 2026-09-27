"""
PHASE 2 — Dataset Analysis, Leakage-Safe Splitting & Feature Engineering

Project:
    SMS Suraksha — Multi-lingual Indian Fraud SMS Detector

Languages:
    English, Hindi, Bengali, Urdu, Punjabi, Marathi

INPUT:
    ../phase1/cleaned_sms_dataset.csv

OUTPUT:
    train.csv
    val.csv
    test.csv
    tfidf_vectorizer.pkl
    feature_summary.txt
    eda_plots/*.png

IMPORTANT:
    The dataset contains translations of the same original SMS.
    Therefore train/validation/test splitting is performed by
    message_id, NOT by individual translated rows.
"""

import os
import re
import string
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


# ==============================================================
# 0. PATHS
# ==============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "..",
    "phase1",
    "cleaned_sms_dataset.csv"
)

OUT_DIR = os.path.join(BASE_DIR, "eda_plots")
os.makedirs(OUT_DIR, exist_ok=True)


TEXT_COL = "text"
LABEL_COL = "label"
LANG_COL = "language"
MESSAGE_ID_COL = "message_id"


# ==============================================================
# 1. LOAD DATASET
# ==============================================================

print("=" * 70)
print("PHASE 2 — DATASET ANALYSIS & FEATURE ENGINEERING")
print("=" * 70)

print(f"\nLoading dataset:")
print(INPUT_FILE)

df = pd.read_csv(INPUT_FILE)

print(f"\nLoaded dataset shape: {df.shape}")
print("\nColumns:")
print(df.columns.tolist())


# ==============================================================
# 2. VALIDATE REQUIRED COLUMNS
# ==============================================================

required_columns = [
    MESSAGE_ID_COL,
    TEXT_COL,
    LABEL_COL,
    LANG_COL
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"\nMissing required columns: {missing_columns}\n"
        f"Available columns: {df.columns.tolist()}"
    )

print("\nRequired columns validated successfully.")


# ==============================================================
# 3. BASIC DATA QUALITY CHECK
# ==============================================================

print("\n" + "=" * 70)
print("DATA QUALITY CHECK")
print("=" * 70)

print("\nMissing values:")
print(df[required_columns].isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nLabels:")
print(df[LABEL_COL].value_counts())

print("\nLanguages:")
print(df[LANG_COL].value_counts())

print("\nUnique original messages:")
print(df[MESSAGE_ID_COL].nunique())


# ==============================================================
# 4. FILTER TO INDIAN LANGUAGES
# ==============================================================

INDIAN_LANGS_CODES = {
    "en",
    "hi",
    "bn",
    "ur",
    "pa",
    "mr"
}

INDIAN_LANGS_NAMES = {
    "english",
    "hindi",
    "bengali",
    "urdu",
    "punjabi",
    "marathi"
}


def is_indian_lang(value):
    value = str(value).strip().lower()

    return (
        value in INDIAN_LANGS_CODES
        or value in INDIAN_LANGS_NAMES
    )


before_filter = len(df)

df = df[
    df[LANG_COL].apply(is_indian_lang)
].copy()

print(
    f"\nLanguage filtering: "
    f"{before_filter} -> {len(df)} rows"
)


# ==============================================================
# 5. NORMALIZE LANGUAGE CODES
# ==============================================================

NAME_TO_CODE = {
    "english": "en",
    "hindi": "hi",
    "bengali": "bn",
    "urdu": "ur",
    "punjabi": "pa",
    "marathi": "mr"
}


df[LANG_COL] = df[LANG_COL].apply(
    lambda value:
    NAME_TO_CODE.get(
        str(value).strip().lower(),
        str(value).strip().lower()
    )
)


print("\nNormalized language distribution:")
print(df[LANG_COL].value_counts())


# ==============================================================
# 6. LABEL NORMALIZATION
# ==============================================================

df[LABEL_COL] = (
    df[LABEL_COL]
    .astype(str)
    .str.strip()
    .str.lower()
)

valid_labels = {"ham", "spam"}

invalid_labels = set(
    df[LABEL_COL].unique()
) - valid_labels

if invalid_labels:
    raise ValueError(
        f"Unexpected labels found: {invalid_labels}"
    )


df["label_binary"] = (
    df[LABEL_COL] == "spam"
).astype(int)


# ==============================================================
# 7. TEXT CLEANUP / VALIDATION
# ==============================================================

df[TEXT_COL] = (
    df[TEXT_COL]
    .fillna("")
    .astype(str)
    .str.strip()
)

before_empty = len(df)

df = df[
    df[TEXT_COL].str.len() > 0
].copy()

print(
    f"\nRemoved empty messages: "
    f"{before_empty - len(df)}"
)


# ==============================================================
# 8. CHECK TRANSLATION GROUPS
# ==============================================================

print("\n" + "=" * 70)
print("TRANSLATION GROUP ANALYSIS")
print("=" * 70)

messages_per_id = (
    df.groupby(MESSAGE_ID_COL)[LANG_COL]
    .nunique()
)

print("\nNumber of languages per original message:")

print(
    messages_per_id.value_counts()
    .sort_index()
)

print(
    f"\nOriginal message IDs: "
    f"{df[MESSAGE_ID_COL].nunique()}"
)

print(
    f"Rows after cleaning: "
    f"{len(df)}"
)


# ==============================================================
# 9. FRAUD / URGENCY KEYWORDS
# ==============================================================

URGENCY_WORDS = {

    "en": [
        "urgent",
        "immediately",
        "verify",
        "suspended",
        "winner",
        "claim",
        "free",
        "act now",
        "limited time",
        "otp",
        "password",
        "click",
        "congratulations",
        "prize",
        "loan",
        "expire",
        "kyc",
        "account blocked",
        "electricity",
        "disconnect",
        "reward"
    ],

    "hi": [
        "तत्काल",
        "तुरंत",
        "सत्यापित",
        "निलंबित",
        "विजेता",
        "दावा",
        "मुफ़्त",
        "पुरस्कार",
        "पासवर्ड",
        "ऋण",
        "केवाईसी",
        "बिजली",
        "कनेक्शन कट",
        "इनाम",
        "लॉटरी"
    ],

    "bn": [
        "জরুরি",
        "অবিলম্বে",
        "যাচাই",
        "স্থগিত",
        "বিজয়ী",
        "বিনামূল্যে",
        "পাসওয়ার্ড",
        "পুরস্কার",
        "ঋণ",
        "কেওয়াইসি",
        "বিদ্যুৎ",
        "লটারি"
    ],

    "ur": [
        "فوری",
        "فوراً",
        "تصدیق",
        "معطل",
        "فاتح",
        "مفت",
        "پاس ورڈ",
        "انعام",
        "قرض",
        "کے وائی سی",
        "بجلی",
        "لاٹری"
    ],

    "pa": [
        "ਜ਼ਰੂਰੀ",
        "ਤੁਰੰਤ",
        "ਤਸਦੀਕ",
        "ਮੁਅੱਤਲ",
        "ਜੇਤੂ",
        "ਮੁਫ਼ਤ",
        "ਪਾਸਵਰਡ",
        "ਇਨਾਮ",
        "ਕਰਜ਼ਾ",
        "ਬਿਜਲੀ",
        "ਲਾਟਰੀ"
    ],

    "mr": [
        "तातडीचे",
        "लगेच",
        "सत्यापित",
        "निलंबित",
        "विजेता",
        "मोफत",
        "पासवर्ड",
        "बक्षीस",
        "कर्ज",
        "वीज",
        "लॉटरी"
    ],

    "default": [
        "urgent",
        "verify",
        "otp",
        "free",
        "winner",
        "password",
        "prize"
    ]
}


# ==============================================================
# 10. REGEX PATTERNS
# ==============================================================

URL_PATTERN = re.compile(
    r"(https?://\S+|www\.\S+|bit\.ly/\S+|tinyurl\S+|t\.co/\S+)",
    re.IGNORECASE
)

PHONE_PATTERN = re.compile(
    r"\b\d{4,}\b"
)

CURRENCY_PATTERN = re.compile(
    r"[₹$]|\b(inr|rs\.?|rupees?)\b",
    re.IGNORECASE
)

EMAIL_PATTERN = re.compile(
    r"[\w\.-]+@[\w\.-]+\.\w+"
)

UPI_PATTERN = re.compile(
    r"\b(upi|paytm|phonepe|gpay|bhim)\b",
    re.IGNORECASE
)


# ==============================================================
# 11. FEATURE FUNCTIONS
# ==============================================================

def urgency_score(text, lang):

    words = URGENCY_WORDS.get(
        lang,
        URGENCY_WORDS["default"]
    )

    text_lower = text.lower()

    return sum(
        1
        for word in words
        if word.lower() in text_lower
    )


def digit_ratio(text):

    if not text:
        return 0.0

    return (
        sum(c.isdigit() for c in text)
        / len(text)
    )


def special_char_ratio(text):

    if not text:
        return 0.0

    return (
        sum(
            1
            for c in text
            if c in string.punctuation
        )
        / len(text)
    )


def caps_ratio(text):

    letters = [
        c
        for c in text
        if c.isalpha()
    ]

    if not letters:
        return 0.0

    return (
        sum(
            1
            for c in letters
            if c.isupper()
        )
        / len(letters)
    )


# ==============================================================
# 12. COMPUTE FRAUD FEATURES
# ==============================================================

print("\nCalculating fraud-related features...")

df["has_url"] = df[TEXT_COL].apply(
    lambda text:
    bool(URL_PATTERN.search(text))
)

df["has_phone_or_code"] = df[TEXT_COL].apply(
    lambda text:
    bool(PHONE_PATTERN.search(text))
)

df["has_currency"] = df[TEXT_COL].apply(
    lambda text:
    bool(CURRENCY_PATTERN.search(text))
)

df["has_email"] = df[TEXT_COL].apply(
    lambda text:
    bool(EMAIL_PATTERN.search(text))
)

df["has_upi_mention"] = df[TEXT_COL].apply(
    lambda text:
    bool(UPI_PATTERN.search(text))
)

df["urgency_score"] = df.apply(
    lambda row:
    urgency_score(
        row[TEXT_COL],
        row[LANG_COL]
    ),
    axis=1
)

df["exclamation_count"] = df[TEXT_COL].apply(
    lambda text:
    text.count("!")
)

df["digit_ratio"] = df[TEXT_COL].apply(
    digit_ratio
)

df["special_char_ratio"] = df[TEXT_COL].apply(
    special_char_ratio
)

df["caps_ratio"] = df[TEXT_COL].apply(
    caps_ratio
)

df["msg_length"] = df[TEXT_COL].apply(
    len
)


df["fraud_signal_count"] = (
    df["has_url"].astype(int)
    + df["has_phone_or_code"].astype(int)
    + df["has_currency"].astype(int)
    + df["has_email"].astype(int)
    + df["has_upi_mention"].astype(int)
    + (df["urgency_score"] >= 1).astype(int)
)


# ==============================================================
# 13. CLASS DISTRIBUTION
# ==============================================================

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

print(
    df[LABEL_COL].value_counts()
)

spam_ratio_overall = (
    df["label_binary"].mean()
)

print(
    f"\nOverall spam ratio: "
    f"{spam_ratio_overall:.4f}"
)


# ==============================================================
# 14. LANGUAGE-SPECIFIC SPAM RATIO
# ==============================================================

spam_ratio = (
    df.groupby(LANG_COL)["label_binary"]
    .mean()
    .sort_values(ascending=False)
)

print("\nSpam ratio by language:")
print(spam_ratio)


# ==============================================================
# 15. CORRELATION ANALYSIS
# ==============================================================

feature_cols = [

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
    "fraud_signal_count"
]


corr = (
    df[
        feature_cols + ["label_binary"]
    ]
    .corr()["label_binary"]
    .drop("label_binary")
    .sort_values(ascending=False)
)

print("\nFeature correlation with spam:")
print(corr)


# ==============================================================
# 16. EDA PLOTS
# ==============================================================

# Language distribution

plt.figure(figsize=(7, 4))

df[LANG_COL].value_counts().plot(
    kind="bar"
)

plt.title(
    "Messages per Indian Language"
)

plt.xlabel("Language")
plt.ylabel("Number of Messages")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT_DIR,
        "indian_lang_distribution.png"
    ),
    dpi=150
)

plt.close()


# Spam ratio

plt.figure(figsize=(7, 4))

spam_ratio.plot(
    kind="bar"
)

plt.title(
    "Spam Ratio by Indian Language"
)

plt.xlabel("Language")
plt.ylabel("Spam Ratio")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT_DIR,
        "indian_lang_spam_ratio.png"
    ),
    dpi=150
)

plt.close()


# Feature correlation

plt.figure(figsize=(8, 5))

corr.plot(
    kind="barh"
)

plt.title(
    "Feature Correlation with Spam Label"
)

plt.xlabel("Correlation")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT_DIR,
        "feature_correlation.png"
    ),
    dpi=150
)

plt.close()


# Message length

plt.figure(figsize=(8, 5))

for label in ["ham", "spam"]:

    subset = df[
        df[LABEL_COL] == label
    ]["msg_length"]

    plt.hist(
        subset,
        bins=40,
        alpha=0.6,
        label=label
    )

plt.xlabel("Message Length")
plt.ylabel("Frequency")
plt.title(
    "Message Length — Ham vs Spam"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT_DIR,
        "message_length_distribution.png"
    ),
    dpi=150
)

plt.close()


# ==============================================================
# 17. LEAKAGE-SAFE TRAIN / VAL / TEST SPLIT
# ==============================================================

print("\n" + "=" * 70)
print("LEAKAGE-SAFE DATA SPLIT")
print("=" * 70)


# One row per ORIGINAL SMS

group_df = (
    df.groupby(MESSAGE_ID_COL)
    .agg(
        label=(
            LABEL_COL,
            "first"
        )
    )
    .reset_index()
)


print(
    f"\nUnique original SMS: "
    f"{len(group_df)}"
)


# First split:
# 70% train
# 30% temporary

train_groups, temp_groups = train_test_split(

    group_df,

    test_size=0.30,

    random_state=42,

    stratify=group_df["label"]
)


# Second split:
# 15% validation
# 15% test

val_groups, test_groups = train_test_split(

    temp_groups,

    test_size=0.50,

    random_state=42,

    stratify=temp_groups["label"]
)


train_ids = set(
    train_groups[MESSAGE_ID_COL]
)

val_ids = set(
    val_groups[MESSAGE_ID_COL]
)

test_ids = set(
    test_groups[MESSAGE_ID_COL]
)


# Safety checks

assert train_ids.isdisjoint(val_ids)
assert train_ids.isdisjoint(test_ids)
assert val_ids.isdisjoint(test_ids)


# Build final row-level datasets

train_df = df[
    df[MESSAGE_ID_COL].isin(train_ids)
].copy()

val_df = df[
    df[MESSAGE_ID_COL].isin(val_ids)
].copy()

test_df = df[
    df[MESSAGE_ID_COL].isin(test_ids)
].copy()


print(
    f"\nTrain rows: "
    f"{train_df.shape}"
)

print(
    f"Validation rows: "
    f"{val_df.shape}"
)

print(
    f"Test rows: "
    f"{test_df.shape}"
)


print("\nOriginal-message groups:")

print(
    f"Train: {len(train_ids)}"
)

print(
    f"Validation: {len(val_ids)}"
)

print(
    f"Test: {len(test_ids)}"
)


# ==============================================================
# 18. VERIFY SPLIT DISTRIBUTIONS
# ==============================================================

print("\nTrain label distribution:")
print(train_df[LABEL_COL].value_counts())

print("\nValidation label distribution:")
print(val_df[LABEL_COL].value_counts())

print("\nTest label distribution:")
print(test_df[LABEL_COL].value_counts())


print("\nTrain language distribution:")
print(train_df[LANG_COL].value_counts())

print("\nValidation language distribution:")
print(val_df[LANG_COL].value_counts())

print("\nTest language distribution:")
print(test_df[LANG_COL].value_counts())


# ==============================================================
# 19. SAVE SPLITS
# ==============================================================

train_df.to_csv(
    os.path.join(BASE_DIR, "train.csv"),
    index=False
)

val_df.to_csv(
    os.path.join(BASE_DIR, "val.csv"),
    index=False
)

test_df.to_csv(
    os.path.join(BASE_DIR, "test.csv"),
    index=False
)


# ==============================================================
# 20. TF-IDF
# ==============================================================
#
# IMPORTANT:
# TF-IDF is FIT ONLY on TRAINING TEXT.
# Validation and test data are transformed later.
#
# This prevents vocabulary leakage.
# ==============================================================

print("\n" + "=" * 70)
print("TF-IDF")
print("=" * 70)

tfidf = TfidfVectorizer(

    analyzer="char_wb",

    ngram_range=(2, 4),

    max_features=20000,

    min_df=2
)


tfidf.fit(
    train_df[TEXT_COL]
)


tfidf_path = os.path.join(
    BASE_DIR,
    "tfidf_vectorizer.pkl"
)

joblib.dump(
    tfidf,
    tfidf_path
)


print(
    f"TF-IDF vocabulary size: "
    f"{len(tfidf.vocabulary_)}"
)

print(
    f"Saved: {tfidf_path}"
)


# ==============================================================
# 21. FEATURE SUMMARY REPORT
# ==============================================================

report_path = os.path.join(
    BASE_DIR,
    "feature_summary.txt"
)


with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "PHASE 2 — FEATURE ENGINEERING SUMMARY\n"
    )

    f.write(
        "=" * 60 + "\n\n"
    )

    f.write(
        "Languages included:\n"
    )

    f.write(
        str(
            sorted(
                df[LANG_COL].unique()
            )
        )
    )

    f.write("\n\n")

    f.write(
        f"Total rows: {len(df)}\n"
    )

    f.write(
        f"Unique original SMS: "
        f"{df[MESSAGE_ID_COL].nunique()}\n\n"
    )

    f.write(
        "Overall class distribution:\n"
    )

    f.write(
        df[LABEL_COL]
        .value_counts()
        .to_string()
    )

    f.write("\n\n")

    f.write(
        "Spam ratio by language:\n"
    )

    f.write(
        spam_ratio.to_string()
    )

    f.write("\n\n")

    f.write(
        "Feature correlation with spam:\n"
    )

    f.write(
        corr.to_string()
    )

    f.write("\n\n")

    f.write(
        "Leakage-safe split:\n"
    )

    f.write(
        f"Train original messages: "
        f"{len(train_ids)}\n"
    )

    f.write(
        f"Validation original messages: "
        f"{len(val_ids)}\n"
    )

    f.write(
        f"Test original messages: "
        f"{len(test_ids)}\n"
    )

    f.write("\n")

    f.write(
        f"Train rows: {len(train_df)}\n"
    )

    f.write(
        f"Validation rows: {len(val_df)}\n"
    )

    f.write(
        f"Test rows: {len(test_df)}\n"
    )

    f.write("\n")

    f.write(
        f"TF-IDF vocabulary size: "
        f"{len(tfidf.vocabulary_)}\n"
    )

    f.write("\n")

    f.write(
        "Fraud-related features:\n"
    )

    for feature in feature_cols:
        f.write(
            f"- {feature}\n"
        )


# ==============================================================
# 22. FINAL OUTPUT
# ==============================================================

print("\n" + "=" * 70)
print("PHASE 2 COMPLETE")
print("=" * 70)

print("\nFiles generated:")

print("  train.csv")
print("  val.csv")
print("  test.csv")
print("  tfidf_vectorizer.pkl")
print("  feature_summary.txt")
print("  eda_plots/")

print("\nIMPORTANT:")
print(
    "Train/validation/test splits were performed by "
    "message_id to prevent translation leakage."
)

print("\nNext phase:")
print(
    "PHASE 3 — TF-IDF Baseline Models"
)

print(
    "  Logistic Regression"
)

print(
    "  Linear SVM"
)

print(
    "  Naive Bayes"
)

print("=" * 70)