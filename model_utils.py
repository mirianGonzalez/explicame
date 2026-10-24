# model_utils.py
import numpy as np
import pandas as pd
import hashlib
import json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import shap

np.random.seed(42)
n = 200
df = pd.DataFrame({
    "ingreso_mensual": np.random.normal(800, 300, n).clip(200, 2500),
    "deuda_actual": np.random.normal(400, 200, n).clip(0, 1500),
    "antiguedad_laboral": np.random.randint(0, 20, n),
    "historial_pago": np.random.choice([0, 1], n, p=[0.3, 0.7]),
})
score = ((df["ingreso_mensual"] > 700).astype(int)
         + (df["deuda_actual"] < 500).astype(int)
         + df["historial_pago"]
         + (df["antiguedad_laboral"] > 5).astype(int))
df["aprobado"] = (score >= 3).astype(int)

X = df.drop(columns=["aprobado"])
y = df["aprobado"]

# --- Modelo 1: Regresión Logística (interpretable) ---
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X, y)
explainer_lr = shap.LinearExplainer(model_lr, X, max_samples=None)

# --- Modelo 2: Random Forest (caja negra, para contraste) ---
model_rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
model_rf.fit(X, y)
explainer_rf = shap.TreeExplainer(model_rf)

MODELOS = {
    "Regresión Logística": {"model": model_lr, "explainer": explainer_lr, "tipo": "interpretable"},
    "Random Forest": {"model": model_rf, "explainer": explainer_rf, "tipo": "caja negra"},
}


def _hash_evidencia(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def _extraer_shap_positivo(shap_vals, n_features: int) -> np.ndarray:
    """Normaliza la salida de SHAP a un vector 1D para la clase positiva."""
    arr = np.array(shap_vals)

    # Lista [clase0, clase1] (versiones antiguas)
    if isinstance(shap_vals, list):
        return np.array(shap_vals[1][0])

    # 3D: (n_samples, n_features, n_clases) -> clase 1
    if arr.ndim == 3:
        return arr[0, :, 1]

    # 2D: (n_samples, n_features)
    if arr.ndim == 2:
        return arr[0, :]

    # 1D: (n_features,)
    if arr.ndim == 1:
        return arr

    raise ValueError(f"Forma de SHAP no reconocida: {arr.shape}")


def evaluar_solicitud(solicitud: dict, nombre_modelo: str = "Regresión Logística"):
    cfg = MODELOS[nombre_modelo]
    model = cfg["model"]
    explainer = cfg["explainer"]

    X_caso = pd.DataFrame([solicitud])
    prob = model.predict_proba(X_caso)[0, 1]
    decision = "APROBADO" if prob >= 0.5 else "RECHAZADO"

    raw_shap = explainer.shap_values(X_caso)
    shap_vals = _extraer_shap_positivo(raw_shap, len(X_caso.columns))

    contribuciones = {col: float(v) for col, v in zip(X_caso.columns, shap_vals)}

    evidencia_base = {
        "decision": decision,
        "probabilidad": round(float(prob), 4),
        "contribuciones_shap": contribuciones,
        "datos_solicitud": solicitud,
    }
    return {
        **evidencia_base,
        "modelo": nombre_modelo,
        "modelo_tipo": cfg["tipo"],
        "version_modelo": "v1.0",
        "fuentes": ["dataset_sintetico_credito_v1", f"modelo_{nombre_modelo.replace(' ', '_').lower()}"],
        "hash_evidencia": _hash_evidencia(evidencia_base),
    }