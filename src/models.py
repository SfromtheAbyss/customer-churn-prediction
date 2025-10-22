import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

def build_pipeline(preprocessor, model_name='xgb'):
    
    if model_name == 'xgb':
        model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )

    elif model_name == 'rf':
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )

    elif model_name == 'logreg':
        model = LogisticRegression(
            solver='lbfgs',
            max_iter=1000,
            random_state=42
        )

    else:
        raise ValueError("Modelo no reconocido. Usa 'xgb', 'rf' o 'logreg'.")

    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    return pipe

def evaluate_model(pipe, X, y, threshold=0.5, verbose=True):
    """
    Evalúa el modelo con métricas de clasificación.
    Devuelve un diccionario con métricas clave.
    """
    probs = pipe.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    metrics = {
        'roc_auc': roc_auc_score(y, probs),
        'precision': precision_score(y, preds),
        'recall': recall_score(y, preds),
        'f1': f1_score(y, preds),
        'confusion_matrix': confusion_matrix(y, preds)
    }

    if verbose:
        print("ROC-AUC:", round(metrics['roc_auc'], 3))
        print("Precision:", round(metrics['precision'], 3))
        print("Recall:", round(metrics['recall'], 3))
        print("F1:", round(metrics['f1'], 3))
        print("\nClassification report:")
        print(classification_report(y, preds))

    return metrics

def save_model(pipe, path='models/best_model.joblib'):
    """Guarda el pipeline entrenado en disco"""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipe, path)
    print(f"✅ Modelo guardado en {path}")