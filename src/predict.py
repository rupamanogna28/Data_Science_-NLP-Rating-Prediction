import pandas as pd
import joblib
import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")

    df_test = pd.read_csv("test.csv")

    df_test["review_text_clean"] = df_test["Review Text"].apply(clean_text)
    df_test["review_title_clean"] = df_test["Review Title"].apply(clean_text)
    df_test["text"] = df_test["review_title_clean"] + " " + df_test["review_text_clean"]

    X_test_vec = vectorizer.transform(df_test["text"])
    predictions = model.predict(X_test_vec)

    submission = pd.DataFrame({
        "id": df_test["id"],
        "Star Rating": predictions
    })

    submission.to_csv("predictions.csv", index=False)
    print("predictions.csv generated ✓")

if __name__ == "__main__":
    predict()
