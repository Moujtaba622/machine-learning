# =============================================
# src/preprocessing.py  (Version Corrigée - Anti-Leakage)
# =============================================

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.utils import (
    load_data, save_data, plot_correlation_heatmap,
    remove_multicollinearity, feature_engineering,
    remove_useless_features, impute_missing_values, apply_pca
)

# ─────────────────────────────────────────────
# LISTE COMPLÈTE des colonnes qui leakent Churn
# (colonnes dérivées de Churn ou post-événement)
# ─────────────────────────────────────────────
LEAKY_COLS = [
    'ChurnRisk', 'AccountStatus', 'CustomerType', 'RFMSegment',
    'Satisfaction', 'SupportTickets', 'LoyaltyLevel',
    'SpendingCat', 'CustomerID', 'Churn',
    # ← ADD THESE:
    'Recency',
    'FavoriteSeason',   # 0.55 post-encoding is too high
    'CustomerTenureDays',  # 0.45 suspicious
    'PreferredMonth',   # 0.43 suspicious
]


def drop_leaky_cols(X: pd.DataFrame) -> pd.DataFrame:
    """Supprime toutes les colonnes leakantes (noms exacts + sous-chaînes)."""
    leaky_keywords = [
        'churn', 'risk', 'accountstatus', 'customertype',
        'perdu', 'closed', 'loyaltylevel', 'rfmsegment',
        'spendingcat', 'satisfaction', 'supportticket'
    ]

    to_drop = set()

    # 1) Correspondances exactes
    for col in LEAKY_COLS:
        if col in X.columns:
            to_drop.add(col)

    # 2) Colonnes encodées (OneHot crée "AccountStatus_Closed", etc.)
    for col in X.columns:
        if any(kw in col.lower() for kw in leaky_keywords):
            to_drop.add(col)

    if to_drop:
        print(f"🛡️  {len(to_drop)} colonnes leakantes supprimées : {sorted(to_drop)}")
        X = X.drop(columns=list(to_drop))
    else:
        print("✅ Aucune colonne leakante détectée.")

    return X


def run_preprocessing():
    print("🚀 Démarrage du pipeline complet (anti-leakage)...\n")

    # ── 1. Chargement ────────────────────────────────────────────
    df = load_data("data/raw/retail.csv")

    # ── 2. Feature Engineering + Parsing ─────────────────────────
    print("\n🔧 Feature Engineering + Parsing...")
    df = feature_engineering(df)

    # ── 3. Séparation X / y  ──────────────────────────────────────
    target_col = "Churn"
    if target_col not in df.columns:
        raise ValueError(f"❌ Colonne '{target_col}' introuvable !")

    y = df[target_col].copy()
    X = df.drop(columns=[target_col])

    # ── 4. SUPPRESSION LEAKAGE (avant tout split) ─────────────────
    print("\n🛡️  Suppression des features leakantes...")
    X = drop_leaky_cols(X)

    # ── 5. Suppression features inutiles ─────────────────────────
    print("\n🧹 Suppression features inutiles (variance nulle / >50% NaN)...")
    X = remove_useless_features(X)

    # ── 6. Train / Test split ─────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n✅ Split → Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"   Distribution Churn (train) : {y_train.value_counts().to_dict()}")

    # Aligner les colonnes test sur train
    X_test = X_test[X_train.columns]

    # ── 7. Imputation ─────────────────────────────────────────────
    print("\n🔄 Imputation des valeurs manquantes (KNN sur train, transform sur test)...")
    X_train, X_test = impute_missing_values(X_train, X_test)

    # ── 8. Corrélation & Multicolinéarité ─────────────────────────
    print("\n🔍 Analyse corrélation & suppression multicolinéarité...")
    _, high_corr = plot_correlation_heatmap(X_train)
    X_train = remove_multicollinearity(X_train, threshold=0.85)
    X_test = X_test[[c for c in X_train.columns if c in X_test.columns]]

    # ── 9. Diagnostic anti-leakage ────────────────────────────────
    print("\n🔍 DIAGNOSTIC : Top 15 corrélations avec Churn (sur X_train numérique)")
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    corr_with_target = X_train[num_cols].corrwith(y_train).abs().sort_values(ascending=False)
    print(corr_with_target.head(15).to_string())
    max_corr = corr_with_target.iloc[0] if len(corr_with_target) > 0 else 0
    if max_corr > 0.7:
        print(f"\n⚠️  ATTENTION : corrélation max = {max_corr:.3f} → leakage potentiel !")
        print(f"   Feature suspecte : {corr_with_target.index[0]}")
    else:
        print(f"\n✅ Corrélation max = {max_corr:.3f} → pas de leakage évident")
    print("-" * 60)

    # ── 10. StandardScaler ────────────────────────────────────────
    print("\n📏 StandardScaler (fit sur train uniquement)...")
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled  = X_test.copy()
    X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test_scaled[numeric_features]  = scaler.transform(X_test[numeric_features])

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")

    # ── 11. ACP ───────────────────────────────────────────────────
    print("\n📉 ACP (sur features numériques scalées)...")
    X_train_pca, X_test_pca, pca_model = apply_pca(
        X_train_scaled[numeric_features],
        X_test_scaled[numeric_features]
    )

    # ── 12. Sauvegarde ────────────────────────────────────────────
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/train_test", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # Fichiers pour train_model.py  (sans scaling, sans PCA)
    X_train.to_csv("data/train_test/X_train.csv", index=False)
    X_test.to_csv("data/train_test/X_test.csv",  index=False)
    y_train.to_csv("data/train_test/y_train.csv", index=False, header=True)
    y_test.to_csv("data/train_test/y_test.csv",   index=False, header=True)

    # Fichiers complémentaires
    save_data(X_train_pca, "data/processed/X_train_pca.csv")
    save_data(X_test_pca,  "data/processed/X_test_pca.csv")

    print("\n🎉 PIPELINE TERMINÉ !")
    print("   → data/train_test/  : prêt pour train_model.py")
    print("   → data/processed/   : versions PCA disponibles")
    print(f"   → Features finales : {X_train.shape[1]} colonnes")

    return X_train_pca, X_test_pca, y_train, y_test, scaler, pca_model


if __name__ == "__main__":
    run_preprocessing()