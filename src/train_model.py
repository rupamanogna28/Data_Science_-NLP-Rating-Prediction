import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score

from data_preprocessing import load_and_prepare_data

def train_and_select_best():
    X_train, X_val, y_train, y_val = load_and_prepare_data()

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        stop_words="english"
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=500),
        "LinearSVM": LinearSVC()
    }

    best_model = None
    best_f1 = 0
    best_name = ""

    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_val_vec)
        f1 = f1_score(y_val, preds, average="weighted")
        print(f"{name} Weighted F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = name

    print(f"\nBest model: {best_name} (F1 = {best_f1:.4f})")

    joblib.dump(best_model, "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")

if __name__ == "__main__":
    train_and_select_best()
