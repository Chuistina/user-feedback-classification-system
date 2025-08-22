import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

def train_and_evaluate_model(input_file, model_dir="models"):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    df = pd.read_csv(input_file)
    
    # For simplicity, let's use the first category if multiple are present
    df["category_single"] = df["category"].apply(lambda x: x.split(", ")[0])
    
    # Filter out categories with less than 2 samples for stratification
    category_counts = df["category_single"].value_counts()
    rare_categories = category_counts[category_counts < 2].index
    df = df[~df["category_single"].isin(rare_categories)]

    X = df["text_for_classification"]
    y = df["category_single"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    # Evaluate
    y_pred = model.predict(X_test_tfidf)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model and vectorizer
    joblib.dump(vectorizer, os.path.join(model_dir, "tfidf_vectorizer.pkl"))
    joblib.dump(model, os.path.join(model_dir, "logistic_regression_model.pkl"))
    print(f"Model and vectorizer saved to {model_dir}")

if __name__ == "__main__":
    input_path = "data/labeling/label_task.csv"
    train_and_evaluate_model(input_path)


