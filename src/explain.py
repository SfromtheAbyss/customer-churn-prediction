import shap
import matplotlib.pyplot as plt

def shap_explain(pipe, X_sample, show_summary=True, n_display=20):
    """Genera explicaciones SHAP para pipeline con XGBoost or tree-based model."""
    # obtener el modelo y el preprocessor
    model = pipe.named_steps['model']
    preproc = pipe.named_steps['preprocessor']

    # transformar X para el modelo
    X_trans = preproc.transform(X_sample)

    if hasattr(model, 'predict_proba') and (model.__class__.__name__.lower().find('xgboost') != -1 or model.__class__.__name__.lower().find('randomforest') != -1):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_trans)
        shap.summary_plot(shap_values, X_trans, show=show_summary, max_display=n_display)
    else:
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_trans, 100))
        shap_values = explainer.shap_values(X_trans)
        shap.summary_plot(shap_values, X_trans, show=show_summary, max_display=n_display)