### This file will contain all text cleaning logic like: lowercase , remove special characters , remove extra spaces , tokenize ,remove stopwords (optional later)
### => Raw Input → Clean Text → ML Model

import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True) # This downloads English stopwords ONCE.

stop_words= set(stopwords.words("english")) # load the English stopwords into a Python set for fast lookup.

# CLEAN TEXT FUNCTION
def clean_text(text: str) -> str:
    """Basic text cleaning for transaction descriptions."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text) # WHAT IT DO "Starbucks Coffee #456!" → "starbucks coffee 456"

    text = re.sub(r"\s+", " ", text).strip()
    return text

# CLEAN + TOKENIZE + REMOVE STOPWORDS FUNCTION
def clean_and_tokenize(text: str):
    cleaned = clean_text(text)
    tokens = [word for word in cleaned.strip() if word not in stop_words]
    return " ".join(tokens)







