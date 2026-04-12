# =============================================
# src/train_model.py  (MODÉLISATION - Version finale corrigée)
# =============================================

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_prepared_data():
    X_train = pd.read_csv("data/train_test/X_train.csv")
    X_test = pd.read_csv("data/train_test/X_test.csv")
    y_train = pd.read_csv("data/train_test/y_train.csv").squeeze()
    y_test = pd.read_csv("data/train_test/y_test.csv").squeeze()
    print(f"✅ Données chargées : {X_train.shape[1]} features | {X_train.shape[0]} clients")
    return X_train, X_test, y_train, y_test

def encode_categorical_features(X_train, X_test):
    """Encodage des variables catégorielles (conforme au sujet)"""
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    
    if not cat_cols:
        return X_train, X_test
    
    print(f"🔄 Encodage de {len(cat_cols)} variables catégorielles...")
    
    # Ordinal pour les variables ordonnées
    ordinal_cols = [col for col in cat_cols if col in ['AgeCategory', 'SpendingCat', 'LoyaltyLevel', 'ChurnRisk', 'BasketSize']]
    onehot_cols = [col for col in cat_cols if col not in ordinal_cols]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), onehot_cols),
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_cols)
        ],
        remainder='passthrough'  # garde les numériques
    )
    
    # Fit uniquement sur train
    X_train_encoded = pd.DataFrame(
        preprocessor.fit_transform(X_train),
        columns=preprocessor.get_feature_names_out(),
        index=X_train.index
    )
    X_test_encoded = pd.DataFrame(
        preprocessor.transform(X_test),
        columns=preprocessor.get_feature_names_out(),
        index=X_test.index
    )
    
    print(f"✅ Encodage terminé → {X_train_encoded.shape[1]} colonnes au total")
    return X_train_encoded, X_test_encoded

def clustering(X_train_pca):
    print("\n🔬 Clustering KMeans (4 segments)...")
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_train_pca)
    
    score = silhouette_score(X_train_pca, clusters)
    print(f"✅ Silhouette Score : {score:.3f}")
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(kmeans, "models/kmeans_model.pkl")
    print("💾 KMeans sauvegardé")
    
    if X_train_pca.shape[1] >= 2:
        plt.figure(figsize=(10, 6))
        plt.scatter(X_train_pca.iloc[:, 0], X_train_pca.iloc[:, 1], c=clusters, cmap='viridis', alpha=0.7)
        plt.title("Segmentation Clients (KMeans + ACP)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.colorbar(label="Cluster")
        plt.savefig("reports/clustering_visualization.png", dpi=300)
        plt.close()
        print("📊 Visualisation clustering sauvegardée")
    else:
        print("⚠️ Visualisation 2D impossible (1 seule composante PCA)")
    
    return kmeans

def classification_models(X_train, X_test, y_train, y_test):
    print("\n🔥 Entraînement RandomForest (prédiction Churn)...")
    
    X_train_enc, X_test_enc = encode_categorical_features(X_train, X_test)
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, None],
        'min_samples_split': [2, 5]
    }
    
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train_enc, y_train)
    
    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(X_test_enc)
    
    print(f"📊 Meilleurs hyperparamètres : {grid_search.best_params_}")
    print("\nRapport de classification :\n")
    print(classification_report(y_test, y_pred, digits=3))
    
    joblib.dump(best_rf, "models/randomforest_churn.pkl")
    print("💾 Meilleur modèle RandomForest sauvegardé")
    
    # Matrice de confusion
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title("Matrice de Confusion - Prédiction Churn")
    plt.ylabel("Vraie valeur")
    plt.xlabel("Prédiction")
    plt.savefig("reports/confusion_matrix.png", dpi=300)
    plt.close()
    
    return best_rf

def run_training():
    X_train, X_test, y_train, y_test = load_prepared_data()
    X_train_pca = pd.read_csv("data/processed/X_train_pca.csv")
    
    clustering(X_train_pca)
    best_model = classification_models(X_train, X_test, y_train, y_test)
    
    print("\n🎉 MODÉLISATION TERMINÉE AVEC SUCCÈS !")
    print("Modèles → models/")
    print("Graphiques → reports/")

if __name__ == "__main__":
    run_training()