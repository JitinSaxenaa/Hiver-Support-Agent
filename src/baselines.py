import os
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from src.taxonomy import get_all_intents

class MajorityClassBaseline:
    def __init__(self):
        self.majority_class: str = 'software_update_os'

    def fit(self, texts: List[str], labels: List[str]):
        series = pd.Series(labels)
        self.majority_class = series.mode()[0]
        return self

    def predict(self, texts: List[str]) -> List[str]:
        return [self.majority_class] * len(texts)

    def predict_single(self, text: str) -> Dict[str, Any]:
        return {
            'intent': self.majority_class,
            'confidence': 1.0,
            'reasoning': f'Majority class prediction ({self.majority_class})'
        }

class TfidfBaselineClassifier:
    def __init__(self, max_features: int = 5000):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=max_features,
            stop_words='english'
        )
        self.classifier = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )
        self.is_fitted = False
        self.classes_ = None

    def fit(self, texts: List[str], labels: List[str]):
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        self.classes_ = list(self.classifier.classes_)
        self.is_fitted = True
        return self

    def predict(self, texts: List[str]) -> List[str]:
        if not self.is_fitted:
            raise RuntimeError('TfidfBaselineClassifier is not fitted yet.')
        X = self.vectorizer.transform(texts)
        return list(self.classifier.predict(X))

    def predict_single(self, text: str) -> Dict[str, Any]:
        if not self.is_fitted:
            raise RuntimeError('TfidfBaselineClassifier is not fitted yet.')
        X = self.vectorizer.transform([text])
        probs = self.classifier.predict_proba(X)[0]
        pred_idx = np.argmax(probs)
        intent = self.classes_[pred_idx]
        conf = float(probs[pred_idx])
        return {
            'intent': intent,
            'confidence': round(conf, 4),
            'reasoning': f'TF-IDF Logistic Regression prediction with probability {conf:.2%}'
        }

    def save(self, path: str = 'models/tfidf_baseline.joblib'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({'vectorizer': self.vectorizer, 'classifier': self.classifier, 'classes': self.classes_}, path)
        print(f'Saved TF-IDF model to {path}')

    def load(self, path: str = 'models/tfidf_baseline.joblib'):
        data = joblib.load(path)
        self.vectorizer = data['vectorizer']
        self.classifier = data['classifier']
        self.classes_ = data['classes']
        self.is_fitted = True
        return self

def evaluate_predictions(y_true: List[str], y_pred: List[str], labels: List[str] = None) -> Dict[str, Any]:
    if labels is None:
        labels = get_all_intents()
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return {
        'accuracy': float(acc),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'labels': labels
    }

def train_and_save_baselines(train_path: str = 'data/processed/train_silver_set.csv') -> Tuple[MajorityClassBaseline, TfidfBaselineClassifier]:
    df_train = pd.read_csv(train_path)
    texts = df_train['inbound_text'].tolist()
    labels = df_train['intent'].tolist()

    majority_model = MajorityClassBaseline().fit(texts, labels)
    tfidf_model = TfidfBaselineClassifier().fit(texts, labels)
    tfidf_model.save('models/tfidf_baseline.joblib')
    return majority_model, tfidf_model

if __name__ == '__main__':
    majority_m, tfidf_m = train_and_save_baselines()
    print(f'Trained Majority Baseline (Class: {majority_m.majority_class})')
    print('Trained TF-IDF Baseline Classifier successfully.')
