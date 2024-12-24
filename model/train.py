import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

def train_model(data_path: str, model_path: str):
    
    # Validate file paths
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The dataset file at {data_path} does not exist.")
    
    # Load dataset
    data = pd.read_csv(data_path, header=0)

    X, y = data['Text'], data['Label']
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Create pipeline
    pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('svm', SVC(kernel='linear', probability=True))
    ])


    # Train model
    pipeline.fit(X_train, y_train)
    # Test model
    y_pred = pipeline.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    # Save model
    joblib.dump(pipeline, model_path)
    print(f"Model saved at {model_path}")

if __name__ == "__main__":
    DATA_PATH = "C:/Users/india/Desktop/text_classification/model/data.csv"
    MODEL_PATH = "C:/Users/india/Desktop/text_classification/model/svm_model.pkl"
    train_model(DATA_PATH, MODEL_PATH)