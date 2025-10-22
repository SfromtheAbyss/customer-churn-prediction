# train.py (ejemplo)
import argparse
from src.data import load_raw, preprocess_basic, train_val_test_split
from src.preprocess import build_preprocessor
from src.models import build_pipeline, save_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

def main(path_csv, out_model='../models/best_model.joblib'):
    df = load_raw(path_csv)
    df = preprocess_basic(df)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df, target='churn')
    preproc, *_ = build_preprocessor(pd.concat([X_train, y_train], axis=1))
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    pipe = build_pipeline(preproc, model)
    pipe.fit(X_train, y_train)
    print("Val ROC-AUC:", roc_auc_score(y_val, pipe.predict_proba(X_val)[:,1]))
    save_model(pipe, out_model)
    print("Modelo guardado en", out_model)

if __name__ == "__main__":
    import fire
    fire.Fire(main)