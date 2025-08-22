from joblib import load

model = load("multi_model.joblib")
vectorizer = load("vectorizer.joblib")
label_binarizer = load("mlb.joblib")  # <- match your actual saved filename


def predict_labels(texts):
    X = vectorizer.transform(texts)
    preds = model.predict(X)
    labels = label_binarizer.inverse_transform(preds)
    return labels
