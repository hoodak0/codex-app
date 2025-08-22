import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump

# Load and prepare data
df = pd.read_csv("codex_sentinel_full_analysis.csv")
df["tag"] = df["tag"].apply(lambda x: [t.strip() for t in x.split(",")])  # convert to list

# Features and labels
X = df["title"]
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df["tag"])

# Vectorization
vectorizer = TfidfVectorizer(max_features=1000)
X_vec = vectorizer.fit_transform(X)

# Model
model = OneVsRestClassifier(LogisticRegression(solver="liblinear"))
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=mlb.classes_))

# Save model and vectorizer
dump(model, "codex_multi_classifier.joblib")  # <- match expected filename)
dump(vectorizer, "vectorizer.joblib")
dump(mlb, "mlb.joblib")
