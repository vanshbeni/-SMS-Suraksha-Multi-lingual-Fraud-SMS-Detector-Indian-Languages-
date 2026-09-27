"""
PHASE 6 — Backend API
Project: SMS Suraksha — Multi-lingual Indian Fraud SMS Detector

FINAL PRODUCTION MODEL:
    Linear SVM from Phase 3

OPTIONAL MODEL:
    MuRIL 10K from Phase 4

The API provides:

    GET  /
    GET  /health
    POST /predict
    POST /predict_batch

The prediction response includes:

    - detected language
    - spam / ham label
    - confidence
    - risk score
    - fraud signals
    - model backend

Project structure expected:

project/
│
├── phase2/
│   ├── tfidf_vectorizer.pkl
│   └── ...
│
├── phase3/
│   └── models/
│       └── linear_svm.pkl
│
├── phase4/
│   └── muril_fraud_model_10k/
│
└── phase6/
    └── backend/
        └── main.py

Run:

    python main.py

OR:

    uvicorn main:app --reload --port 8000
"""

# ==============================================================
# IMPORTS
# ==============================================================

import os
import re
import string
from typing import List

import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ==============================================================
# PATHS
# ==============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "..",
        ".."
    )
)

PHASE2_DIR = os.path.join(
    PROJECT_DIR,
    "phase2"
)

PHASE3_DIR = os.path.join(
    PROJECT_DIR,
    "phase3"
)

PHASE4_DIR = os.path.join(
    PROJECT_DIR,
    "phase4"
)


TFIDF_PATH = os.path.join(
    PHASE2_DIR,
    "tfidf_vectorizer.pkl"
)

SVM_PATH = os.path.join(
    PHASE3_DIR,
    "models",
    "linear_svm.pkl"
)

MURIL_PATH = os.path.join(
    PHASE4_DIR,
    "muril_fraud_model_10k"
)


# ==============================================================
# FASTAPI
# ==============================================================

app = FastAPI(
    title="SMS Suraksha API",
    description=(
        "Multi-lingual Indian SMS Fraud Detection API"
    ),
    version="1.0.0",
)


# ==============================================================
# CORS
# ==============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ==============================================================
# FRAUD / URGENCY WORDS
# Same conceptual feature set used in Phase 2
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
        "reward",
    ],

    "hi": [
        "तत्काल",
        "तुरंत",
        "सत्यापित",
        "निलंबित",
        "विजेता",
        "मुफ़्त",
        "पासवर्ड",
        "ऋण",
        "केवाईसी",
        "बिजली",
        "इनाम",
        "लॉटरी",
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
        "লটারি",
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
        "لاٹری",
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
        "ਲਾਟਰੀ",
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
        "लॉटरी",
    ],

    "default": [
        "urgent",
        "verify",
        "otp",
        "free",
        "winner",
        "password",
        "prize",
    ],
}


# ==============================================================
# REGEX PATTERNS
# ==============================================================

URL_PATTERN = re.compile(
    r"(https?://\S+|www\.\S+|bit\.ly/\S+|tinyurl\S+|t\.co/\S+)",
    re.IGNORECASE,
)

PHONE_PATTERN = re.compile(
    r"\b\d{4,}\b"
)

CURRENCY_PATTERN = re.compile(
    r"[₹$]|\b(inr|rs\.?|rupees?)\b",
    re.IGNORECASE,
)

EMAIL_PATTERN = re.compile(
    r"[\w\.-]+@[\w\.-]+\.\w+"
)

UPI_PATTERN = re.compile(
    r"\b(upi|paytm|phonepe|gpay|bhim)\b",
    re.IGNORECASE,
)


# ==============================================================
# LANGUAGE DETECTION
# ==============================================================

def detect_language(text: str) -> str:

    try:

        from langdetect import detect

        detected = detect(text)

        supported = {
            "en",
            "hi",
            "bn",
            "ur",
            "pa",
            "mr",
        }

        if detected in supported:
            return detected

        return "default"

    except Exception:

        return "default"


# ==============================================================
# FRAUD SIGNAL DETECTION
# ==============================================================

def get_fraud_signals(
    text: str,
    lang: str
):

    words = URGENCY_WORDS.get(
        lang,
        URGENCY_WORDS["default"]
    )

    text_lower = text.lower()

    signals = []


    # URL
    if URL_PATTERN.search(text):

        signals.append({
            "tag": "Suspicious link",
            "icon": "🔗",
        })


    # UPI
    if UPI_PATTERN.search(text):

        signals.append({
            "tag": "Payment app mention",
            "icon": "💳",
        })


    # Currency
    if CURRENCY_PATTERN.search(text):

        signals.append({
            "tag": "Money amount mentioned",
            "icon": "💰",
        })


    # Email
    if EMAIL_PATTERN.search(text):

        signals.append({
            "tag": "Email address present",
            "icon": "✉️",
        })


    # Phone / code
    if PHONE_PATTERN.search(text):

        signals.append({
            "tag": "Number/code present",
            "icon": "🔢",
        })


    # Urgency
    urgency_hits = [
        word
        for word in words
        if word.lower() in text_lower
    ]


    if urgency_hits:

        signals.append({
            "tag": "Urgency / scam language",
            "icon": "⏰",
        })


    return signals


# ==============================================================
# FEATURE ENGINEERING
# Same 12 features used by Phase 3
# ==============================================================

def digit_ratio(text: str):

    if not text:
        return 0.0

    return (
        sum(
            character.isdigit()
            for character in text
        )
        / len(text)
    )


def special_char_ratio(text: str):

    if not text:
        return 0.0

    return (
        sum(
            character in string.punctuation
            for character in text
        )
        / len(text)
    )


def caps_ratio(text: str):

    letters = [
        character
        for character in text
        if character.isalpha()
    ]

    if not letters:
        return 0.0

    return (
        sum(
            character.isupper()
            for character in letters
        )
        / len(letters)
    )


def urgency_score(
    text: str,
    lang: str
):

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


def build_features(
    text: str,
    lang: str
):

    has_url = int(
        bool(
            URL_PATTERN.search(text)
        )
    )

    has_phone_or_code = int(
        bool(
            PHONE_PATTERN.search(text)
        )
    )

    has_currency = int(
        bool(
            CURRENCY_PATTERN.search(text)
        )
    )

    has_email = int(
        bool(
            EMAIL_PATTERN.search(text)
        )
    )

    has_upi_mention = int(
        bool(
            UPI_PATTERN.search(text)
        )
    )

    urgency = urgency_score(
        text,
        lang
    )

    exclamation_count = text.count(
        "!"
    )

    digits = digit_ratio(
        text
    )

    special = special_char_ratio(
        text
    )

    caps = caps_ratio(
        text
    )

    message_length = len(
        text
    )

    fraud_signal_count = (
        has_url
        + has_phone_or_code
        + has_currency
        + has_email
        + has_upi_mention
        + int(urgency >= 1)
    )


    return np.array(
        [[
            has_url,
            has_phone_or_code,
            has_currency,
            has_email,
            has_upi_mention,
            urgency,
            exclamation_count,
            digits,
            special,
            caps,
            message_length,
            fraud_signal_count,
        ]],
        dtype=float,
    )


# ==============================================================
# MODEL LOADING
# ==============================================================

MODEL_BACKEND = None

_tfidf = None
_svm = None

_muril_tokenizer = None
_muril_model = None
_torch = None


print("=" * 70)
print("SMS SURAKSHA — PHASE 6 BACKEND")
print("=" * 70)

print(
    "\nProject directory:"
)

print(
    PROJECT_DIR
)


# ==============================================================
# LOAD LINEAR SVM
# ==============================================================
#
# This is the PRIMARY production model because:
#
# SVM F1     = 0.9295
# MuRIL F1   = 0.9119
#
# ==============================================================

try:

    import joblib
    from scipy.sparse import hstack, csr_matrix

    print(
        "\nLoading TF-IDF vectorizer..."
    )

    _tfidf = joblib.load(
        TFIDF_PATH
    )


    print(
        "Loading Linear SVM..."
    )

    _svm = joblib.load(
        SVM_PATH
    )


    MODEL_BACKEND = (
        "linear_svm"
    )


    print(
        "\nLoaded final production model:"
    )

    print(
        "Linear SVM + TF-IDF + fraud features"
    )


except Exception as e:

    print(
        "\nLinear SVM could not be loaded:"
    )

    print(
        str(e)
    )


# ==============================================================
# OPTIONAL MURIL LOADER
# ==============================================================

if os.path.isdir(
    MURIL_PATH
):

    try:

        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
        )


        print(
            "\nMuRIL 10K model is available."
        )

        print(
            "It will remain available as an optional "
            "transformer backend."
        )


        _muril_tokenizer = (
            AutoTokenizer.from_pretrained(
                MURIL_PATH
            )
        )


        _muril_model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                MURIL_PATH
            )
        )


        _muril_model.eval()


        if torch.cuda.is_available():

            _muril_model = (
                _muril_model.cuda()
            )


        _torch = torch


        print(
            "MuRIL 10K loaded successfully."
        )


    except Exception as e:

        print(
            "\nMuRIL optional loading failed:"
        )

        print(
            str(e)
        )

else:

    print(
        "\nMuRIL 10K model not found."
    )

    print(
        "This is okay because Linear SVM "
        "is the production model."
    )


# ==============================================================
# VERIFY PRIMARY MODEL
# ==============================================================

if MODEL_BACKEND is None:

    raise RuntimeError(
        "\nNo production model could be loaded.\n\n"
        "Expected:\n"
        f"{TFIDF_PATH}\n"
        f"{SVM_PATH}\n"
    )


# ==============================================================
# PREDICTION — LINEAR SVM
# ==============================================================

def predict_svm(
    text: str,
    lang: str
):

    from scipy.sparse import (
        hstack,
        csr_matrix,
    )


    # TF-IDF
    tfidf_vector = (
        _tfidf.transform(
            [text]
        )
    )


    # Phase 2 numeric features
    extra_features = (
        build_features(
            text,
            lang
        )
    )


    extra_sparse = csr_matrix(
        extra_features
    )


    # Combine exactly like Phase 3
    full_vector = hstack(
        [
            tfidf_vector,
            extra_sparse,
        ]
    )


    # Linear SVM prediction
    prediction = int(
        _svm.predict(
            full_vector
        )[0]
    )


    # LinearSVC does not have predict_proba.
    # decision_function gives a signed score.
    decision = float(
        _svm.decision_function(
            full_vector
        )[0]
    )


    # Convert decision score to a smooth
    # probability-like confidence score.
    #
    # This is NOT a calibrated probability.
    # It is used only for the UI risk meter.
    spam_score = (
        1.0
        / (
            1.0
            + np.exp(
                -decision
            )
        )
    )


    return (
        prediction,
        float(spam_score),
        decision,
    )


# ==============================================================
# OPTIONAL MURIL PREDICTION
# ==============================================================

def predict_muril(
    text: str
):

    if (
        _muril_model is None
        or _muril_tokenizer is None
        or _torch is None
    ):

        raise RuntimeError(
            "MuRIL model is not loaded."
        )


    device = next(
        _muril_model.parameters()
    ).device


    with _torch.no_grad():

        encoded = (
            _muril_tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=64,
                return_tensors="pt",
            )
        )


        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }


        logits = (
            _muril_model(
                **encoded
            ).logits
        )


        probabilities = (
            _torch.softmax(
                logits,
                dim=1
            )
            .cpu()
            .numpy()[0]
        )


    spam_probability = float(
        probabilities[1]
    )


    prediction = int(
        spam_probability >= 0.5
    )


    return (
        prediction,
        spam_probability,
    )


# ==============================================================
# MAIN PREDICTION FUNCTION
# ==============================================================

def predict(
    text: str,
    lang: str
):

    if MODEL_BACKEND == "linear_svm":

        prediction, spam_score, decision = (
            predict_svm(
                text,
                lang
            )
        )


        return {
            "prediction": prediction,
            "spam_probability": spam_score,
            "decision_score": decision,
            "backend": "linear_svm",
        }


    raise RuntimeError(
        "No valid prediction backend."
    )


# ==============================================================
# API SCHEMAS
# ==============================================================

class SMSRequest(
    BaseModel
):

    text: str


class SMSResponse(
    BaseModel
):

    text: str

    language_detected: str

    label: str

    confidence: float

    risk_score: int

    signals: List[dict]

    model_backend: str


class BatchRequest(
    BaseModel
):

    messages: List[str]


# ==============================================================
# ROOT
# ==============================================================

@app.get("/")
def root():

    return {

        "status": "ok",

        "service": (
            "SMS Suraksha "
            "Indian Multi-lingual "
            "Fraud Detector"
        ),

        "model_backend": (
            MODEL_BACKEND
        ),

        "muril_available": (
            _muril_model is not None
        ),
    }


# ==============================================================
# HEALTH CHECK
# ==============================================================

@app.get("/health")
def health():

    return {

        "status": "healthy",

        "model_backend": (
            MODEL_BACKEND
        ),

        "linear_svm_loaded": (
            _svm is not None
        ),

        "tfidf_loaded": (
            _tfidf is not None
        ),

        "muril_loaded": (
            _muril_model is not None
        ),
    }


# ==============================================================
# PREDICT
# ==============================================================

@app.post(
    "/predict",
    response_model=SMSResponse
)
def predict_sms(
    req: SMSRequest
):

    text = req.text.strip()


    if not text:

        raise HTTPException(
            status_code=400,
            detail=(
                "Empty message text."
            ),
        )


    # Detect language
    language = detect_language(
        text
    )


    # Model prediction
    result = predict(
        text,
        language
    )


    prediction = result[
        "prediction"
    ]


    spam_probability = result[
        "spam_probability"
    ]


    # Label
    label = (
        "spam"
        if prediction == 1
        else "ham"
    )


    # Confidence
    #
    # For SVM this is a probability-like
    # score derived from the decision function,
    # not a calibrated probability.
    confidence = (
        spam_probability
        if label == "spam"
        else 1.0 - spam_probability
    )


    confidence = max(
        0.0,
        min(
            1.0,
            confidence
        )
    )


    # Risk score
    risk_score = int(
        round(
            spam_probability * 100
        )
    )


    # Fraud signals
    signals = get_fraud_signals(
        text,
        language
    )


    return SMSResponse(

        text=text,

        language_detected=language,

        label=label,

        confidence=round(
            confidence,
            4
        ),

        risk_score=risk_score,

        signals=signals,

        model_backend=MODEL_BACKEND,
    )


# ==============================================================
# BATCH PREDICTION
# ==============================================================

@app.post(
    "/predict_batch"
)
def predict_batch(
    req: BatchRequest
):

    if not req.messages:

        raise HTTPException(
            status_code=400,
            detail=(
                "No messages provided."
            ),
        )


    results = []


    for message in req.messages:

        response = predict_sms(
            SMSRequest(
                text=message
            )
        )


        results.append(
            response.model_dump()
        )


    return results


# ==============================================================
# RUN DIRECTLY
# ==============================================================

if __name__ == "__main__":

    import uvicorn

    print(
        "\nStarting SMS Suraksha API..."
    )

    print(
        "API: http://127.0.0.1:8000"
    )

    print(
        "Docs: http://127.0.0.1:8000/docs"
    )


    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
    )