# =============================================
# src/predict.py  (Prédiction en production)
# =============================================

import pandas as pd
import joblib
import os

def load_models():
    """Charge les modèles entraînés"""
    kmeans = joblib.load("models/kmeans_model.pkl")
    rf = joblib.load("models/randomforest_churn.pkl")
    return kmeans, rf

def predict_client(client_data: dict):
    """Prédit le churn et le segment d'un nouveau client"""
    df = pd.DataFrame([client_data])
    
    # On charge le preprocessor si besoin (mais ici on utilise le modèle déjà entraîné)
    kmeans, rf = load_models()
    
    # Encodage rapide (même que dans train_model)
    # (on suppose que tu envoies déjà les features encodées ou on peut le faire ici)
    prediction_churn = rf.predict(df)[0]
    segment = kmeans.predict(df)[0]
    
    result = {
        "Churn_Prediction": "Parti (risque élevé)" if prediction_churn == 1 else "Fidèle",
        "Client_Segment": segment,
        "Recommendation": "Campagne fidélisation" if prediction_churn == 0 else "Offre spéciale pour récupérer le client"
    }
    return result

# Test rapide
if __name__ == "__main__":
    # Exemple de client (à adapter)
    example = {
        "Recency": 5,
        "Frequency": 12,
        "MonetaryTotal": 1250.75,
        # ... (ajoute d'autres features selon tes colonnes)
        "CustomerTenure": 180
    }
    print(predict_client(example))