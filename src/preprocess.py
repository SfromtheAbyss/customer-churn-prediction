from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd
import sklearn
import category_encoders as ce


def get_onehot_encoder():
    """Garantiza compatibilidad entre versiones de sklearn."""
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown='ignore', sparse=False)


def build_preprocessor(df: pd.DataFrame,
                       numeric_features=None,
                       categorical_features=None):
    """
    Construye un ColumnTransformer robusto para preprocesamiento.

    - Imputa valores faltantes.
    - Escala numéricos.
    - Codifica categóricos (OneHot o Target Encoding según cardinalidad).

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame con los datos (sin target si es posible).
    numeric_features : list, opcional
        Columnas numéricas.
    categorical_features : list, opcional
        Columnas categóricas.

    Retorna:
    --------
    preprocessor : ColumnTransformer
        Pipeline de preprocesamiento completo.
    numeric_features : list
        Columnas numéricas.
    low_card : list
        Categóricas de baja cardinalidad.
    high_card : list
        Categóricas de alta cardinalidad.
    """

    # Inferencia de tipos si no se pasan manualmente
    if numeric_features is None:
        numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if categorical_features is None:
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Evitar incluir la columna target por error
    possible_targets = ['churn', 'target', 'label']
    numeric_features = [c for c in numeric_features if c.lower() not in possible_targets]
    categorical_features = [c for c in categorical_features if c.lower() not in possible_targets]

    # Pipelines individuales
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    low_card = [c for c in categorical_features if df[c].nunique() <= 10]
    high_card = [c for c in categorical_features if df[c].nunique() > 10]

    cat_low_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', get_onehot_encoder())
    ])

    cat_high_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('target_enc', ce.TargetEncoder())  # se entrena junto al modelo
    ])

    transformers = []
    if numeric_features:
        transformers.append(('num', numeric_transformer, numeric_features))
    if low_card:
        transformers.append(('cat_low', cat_low_transformer, low_card))
    if high_card:
        transformers.append(('cat_high', cat_high_transformer, high_card))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop',
        sparse_threshold=0
    )

    return preprocessor, numeric_features, low_card, high_card