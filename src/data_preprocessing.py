import re
import pandas as pd
from sklearn.model_selection import train_test_split

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_and_prepare_data(path="train.csv"):
    df = pd.read_csv(path)

    # Clean text fields
    df["review_text_clean"] = df["Review Text"].apply(clean_text)
    df["review_title_clean"] = df["Review Title"].apply(clean_text)

    # Combine title + review
    df["text"] = df["review_title_clean"] + " " + df["review_text_clean"]

    X = df["text"]
    y = df["Star Rating"]

    # Stratified split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_val, y_train, y_val
