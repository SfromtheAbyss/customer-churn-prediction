import pandas as pd
from sklearn.model_selection import train_test_split

def load_raw(path: str) -> pd.DataFrame:
    """Carga CSV crudo"""
    return pd.read_csv(path)

def preprocess_basic(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza ligera de columnas típicas del dataset Telco"""
    df = df.copy()
    # Ejemplos: convertir total_charges a num, borrar id, strip espacios
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # Rename columns to snake_case
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    return df

def train_val_test_split(df: pd.DataFrame, target: str='churn', test_size=0.2, val_size=0.1, random_state=42):
    """Split train/val/test stratificado por target"""
    X = df.drop(columns=[target])
    y = df[target].map({'Yes':1, 'No':0}) if df[target].dtype == object else df[target]
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    val_relative = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=val_relative, stratify=y_trainval, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test