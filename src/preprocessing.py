# =============================================
# src/preprocessing.py  (PIPELINE COMPLET - Sections 1 à 7)
# =============================================

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.utils import (
    load_data, save_data, plot_correlation_heatmap,
    remove_multicollinearity, feature_engineering,
    remove_useless_features, impute_missing_values, apply_pca
)

def run_preprocessing():
    print("🚀 Démarrage du pipeline complet (sections 1 à 7)...")
    
    df = load_data("data/raw/retail.csv")         
    
    print("\n🔧 Feature Engineering + Parsing...")
    df = feature_engineering(df)
    
    target_col = "Churn"
    if target_col not in df.columns:
        raise ValueError(f" Colonne '{target_col}' introuvable !")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("\n🧹 Suppression features inutiles...")
    X_train = remove_useless_features(X_train)
    X_test = X_test[X_train.columns]
    
    print("\n🔄 Imputation des valeurs manquantes...")
    X_train, X_test = impute_missing_values(X_train, X_test)
    
    print("\n📏 StandardScaler...")
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    if "CustomerID" in numeric_features:
        numeric_features.remove("CustomerID")
    
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])
    
    print("\n🔍 Corrélation & multicolinéarité...")
    _, _ = plot_correlation_heatmap(X_train)
    X_train = remove_multicollinearity(X_train)
    X_test = X_test[X_train.columns]
    
    print("\n📉 ACP...")
    numeric_cols = X_train_scaled.select_dtypes(include=[np.number]).columns.tolist()
    X_train_numeric = X_train_scaled[numeric_cols]
    X_test_numeric = X_test_scaled[numeric_cols]
    X_train_pca, X_test_pca, pca_model = apply_pca(X_train_numeric, X_test_numeric)
    
    # Sauvegarde
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/train_test", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    save_data(X_train, "data/processed/X_train.csv")
    save_data(X_test, "data/processed/X_test.csv")
    save_data(pd.DataFrame(y_train, columns=[target_col]), "data/processed/y_train.csv")
    save_data(pd.DataFrame(y_test, columns=[target_col]), "data/processed/y_test.csv")
    
    save_data(X_train_pca, "data/processed/X_train_pca.csv")
    save_data(X_test_pca, "data/processed/X_test_pca.csv")
    
    X_train.to_csv("data/train_test/X_train.csv", index=False)
    X_test.to_csv("data/train_test/X_test.csv", index=False)
    pd.DataFrame(y_train, columns=[target_col]).to_csv("data/train_test/y_train.csv", index=False)
    pd.DataFrame(y_test, columns=[target_col]).to_csv("data/train_test/y_test.csv", index=False)
    
    print("\n🎉 PIPELINE COMPLET TERMINÉ (sections 1 à 7) !")
    print("Fichiers prêts dans data/processed/ et data/train_test/")
    return X_train_pca, X_test_pca, y_train, y_test, scaler, pca_model

if __name__ == "__main__":
    run_preprocessing()