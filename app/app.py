# =============================================
# app/app.py  — Interface Flask (Déploiement ML)
# =============================================

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

app = Flask(__name__)

# ── Chargement des modèles et de l'encodeur ─────────────────────
def load_artifacts():
    artifacts = {}
    paths = {
        "rf":      "models/randomforest_churn.pkl",
        "kmeans":  "models/kmeans_model.pkl",
        "scaler":  "models/scaler.pkl",
        "encoder": "models/encoder.pkl",        # encodeur sauvegardé par train_model.py
    }
    for name, path in paths.items():
        if os.path.exists(path):
            artifacts[name] = joblib.load(path)
        else:
            print(f"⚠️  {path} introuvable — certaines fonctions seront limitées")
    return artifacts

ARTIFACTS = load_artifacts()

# Colonnes après encodage (telles que vues par le modèle à l'entraînement)
try:
    MODEL_FEATURE_NAMES = ARTIFACTS["rf"].feature_names_in_.tolist()
except Exception:
    MODEL_FEATURE_NAMES = []

LEAKY_KEYWORDS = [
    'churn', 'risk', 'accountstatus', 'customertype', 'perdu', 'closed',
    'loyaltylevel', 'rfmsegment', 'spendingcat', 'satisfaction',
    'supportticket', 'customerid', 'recency', 'favoriteseason',
    'customertenuredays', 'preferredmonth'
]

SEGMENT_NAMES = {
    0: ("Champions",  "Acheteurs fréquents, haute valeur",      "#22c55e"),
    1: ("Fidèles",    "Réguliers, bon potentiel de rétention",  "#3b82f6"),
    2: ("Potentiels", "Actifs mais pas encore réguliers",       "#f59e0b"),
    3: ("Dormants",   "Inactifs, risque de départ élevé",       "#ef4444"),
}


def encode_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode un DataFrame brut exactement comme train_model.py le fait,
    en utilisant l'encodeur sauvegardé (models/encoder.pkl).
    Si l'encodeur n'est pas disponible, fallback sur pd.get_dummies.
    """
    encoder = ARTIFACTS.get("encoder")

    if encoder is not None:
        # Utiliser le même ColumnTransformer que lors de l'entraînement
        try:
            encoded = pd.DataFrame(
                encoder.transform(df),
                columns=encoder.get_feature_names_out(),
                index=df.index
            )
            return encoded
        except Exception as e:
            print(f"⚠️  Encodeur échoué ({e}), fallback get_dummies")

    # Fallback : get_dummies simple
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols)
    return df


def preprocess_for_model(data: dict) -> pd.DataFrame:
    """
    Pipeline complet : dict brut → DataFrame aligné sur les colonnes du modèle.
    """
    df = pd.DataFrame([data])

    # Convertir les numériques
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass

    # Supprimer les colonnes leakantes
    to_drop = [c for c in df.columns if any(kw in c.lower() for kw in LEAKY_KEYWORDS)]
    df = df.drop(columns=to_drop, errors='ignore')

    # Encoder
    df = encode_dataframe(df)

    # Aligner sur les features du modèle
    if MODEL_FEATURE_NAMES:
        for col in MODEL_FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0
        df = df[MODEL_FEATURE_NAMES]

    return df.fillna(0)


def preprocess_test_set(X_test: pd.DataFrame) -> pd.DataFrame:
    """
    Encode X_test.csv (colonnes brutes) pour le dashboard.
    Même logique que preprocess_for_model mais sur un DataFrame entier.
    """
    # Supprimer leaky
    to_drop = [c for c in X_test.columns if any(kw in c.lower() for kw in LEAKY_KEYWORDS)]
    X_test = X_test.drop(columns=to_drop, errors='ignore')

    # Encoder
    X_test = encode_dataframe(X_test)

    # Aligner
    if MODEL_FEATURE_NAMES:
        for col in MODEL_FEATURE_NAMES:
            if col not in X_test.columns:
                X_test[col] = 0
        X_test = X_test[MODEL_FEATURE_NAMES]

    return X_test.fillna(0)


def risk_level(proba: float):
    if proba >= 0.75: return ("Critique", "#ef4444")
    if proba >= 0.50: return ("Élevé",    "#f97316")
    if proba >= 0.25: return ("Moyen",    "#f59e0b")
    return                    ("Faible",   "#22c55e")


# ── Routes ──────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html", result=None)

    try:
        form_data = request.form.to_dict()
        df = preprocess_for_model(form_data)

        rf = ARTIFACTS.get("rf")
        if rf is None:
            return render_template("predict.html",
                                   error="Modèle RandomForest non chargé. Lancez src/train_model.py d'abord.",
                                   result=None)

        pred  = int(rf.predict(df)[0])
        proba = float(rf.predict_proba(df)[0][1])
        risk, risk_color = risk_level(proba)

        segment_id   = None
        segment_info = None
        kmeans = ARTIFACTS.get("kmeans")
        if kmeans is not None:
            try:
                num_data    = df.select_dtypes(include=[np.number])
                segment_id  = int(kmeans.predict(num_data)[0])
                segment_info = SEGMENT_NAMES.get(segment_id, (f"Segment {segment_id}", "", "#6b7280"))
            except Exception:
                pass

        result = {
            "prediction":    pred,
            "label":         "Churné" if pred == 1 else "Fidèle",
            "probability":   round(proba * 100, 1),
            "risk":          risk,
            "risk_color":    risk_color,
            "segment_id":    segment_id,
            "segment_name":  segment_info[0] if segment_info else None,
            "segment_desc":  segment_info[1] if segment_info else None,
            "segment_color": segment_info[2] if segment_info else "#6b7280",
        }
        return render_template("predict.html", result=result, form=form_data)

    except Exception as e:
        return render_template("predict.html", error=f"Erreur : {str(e)}", result=None)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data  = request.get_json(force=True)
        df    = preprocess_for_model(data)
        rf    = ARTIFACTS.get("rf")
        if rf is None:
            return jsonify({"error": "Modèle non chargé"}), 500
        pred  = int(rf.predict(df)[0])
        proba = float(rf.predict_proba(df)[0][1])
        risk, _ = risk_level(proba)
        return jsonify({
            "churn_prediction":  pred,
            "churn_label":       "Churné" if pred == 1 else "Fidèle",
            "churn_probability": round(proba, 4),
            "risk_level":        risk
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/dashboard")
def dashboard():
    stats = {}
    try:
        y_test = pd.read_csv("data/train_test/y_test.csv").squeeze()
        X_test = pd.read_csv("data/train_test/X_test.csv")
        rf     = ARTIFACTS.get("rf")

        if rf is None:
            stats["error"] = "Modèle non chargé. Lancez src/train_model.py d'abord."
            return render_template("dashboard.html", stats=stats)

        # Encoder X_test exactement comme le modèle l'attend
        X_enc  = preprocess_test_set(X_test)
        preds  = rf.predict(X_enc)
        probas = rf.predict_proba(X_enc)[:, 1]

        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        stats = {
            "accuracy":  round(accuracy_score(y_test, preds) * 100, 1),
            "f1":        round(f1_score(y_test, preds) * 100, 1),
            "precision": round(precision_score(y_test, preds) * 100, 1),
            "recall":    round(recall_score(y_test, preds) * 100, 1),
            "total":     len(y_test),
            "churners":  int(preds.sum()),
            "rate":      round(preds.mean() * 100, 1),
            "high_risk": int((probas >= 0.75).sum()),
        }

    except Exception as e:
        stats["error"] = str(e)

    return render_template("dashboard.html", stats=stats)


if __name__ == "__main__":
    print("🚀 Lancement de l'application Flask → http://127.0.0.1:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)