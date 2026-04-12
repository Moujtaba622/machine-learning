
import pandas as pd
import numpy as np
import os
import joblib
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import Tuple, Optional

def load_data(file_path: str = "data/raw/retail.csv") -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Fichier non trouvé : {file_path}\nVérifiez que votre CSV est bien dans data/raw/")
    df = pd.read_csv(file_path)
    print(f"✅ Données chargées → {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
    return df

def save_data(df: pd.DataFrame, file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    print(f"💾 Sauvegardé → {file_path}")

def plot_correlation_heatmap(df: pd.DataFrame, threshold: float = 0.8, 
                             save_path: str = "reports/correlation_heatmap.png") -> Tuple[pd.DataFrame, list]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(22, 18))
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", center=0, linewidths=0.5, square=True)
    plt.title("Matrice de Corrélation - Features Numériques", fontsize=16)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📊 Heatmap sauvegardée → {save_path}")
    
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], round(corr_matrix.iloc[i, j], 3)))
    if high_corr:
        print(f"⚠️ {len(high_corr)} paires fortement corrélées (|corr| > {threshold})")
    return corr_matrix, high_corr

def remove_multicollinearity(df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = df[numeric_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    if to_drop:
        print(f"🗑️ Features supprimées pour multicolinéarité : {to_drop}")
        df = df.drop(columns=to_drop)
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """4. Feature Engineering + 7. Parsing (RegistrationDate + LastLoginIP)"""
    df = df.copy()
    
    # Ratios métier
    if 'MonetaryTotal' in df.columns and 'Recency' in df.columns:
        df['MonetaryPerDay'] = df['MonetaryTotal'] / (df['Recency'] + 1)
    if 'MonetaryTotal' in df.columns and 'Frequency' in df.columns:
        df['AvgBasketValue'] = df['MonetaryTotal'] / df['Frequency']
    if 'Recency' in df.columns and 'CustomerTenure' in df.columns:
        df['TenureRatio'] = df['Recency'] / df['CustomerTenure'].replace(0, np.nan)
    
    # Parsing RegistrationDate
    date_col = next((col for col in ['RegistrationDate', 'RegistDate'] if col in df.columns), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
        df['RegYear'] = df[date_col].dt.year
        df['RegMonth'] = df[date_col].dt.month
        df['RegDay'] = df[date_col].dt.day
        df['RegWeekday'] = df[date_col].dt.weekday
        df = df.drop(columns=[date_col])          # ← FIX IMPORTANT : on supprime la colonne datetime
        print(f"✅ {date_col} parsé et supprimé (features extraites : RegYear, RegMonth...)")
    
    # Parsing LastLoginIP
    if 'LastLoginIP' in df.columns:
        def extract_ip(ip):
            if pd.isna(ip): 
                return pd.Series({'IsPrivateIP': np.nan, 'IPVersion': np.nan})
            ip_str = str(ip).strip()
            private = 1 if re.match(r'^(10\.|172\.(1[6-9]|2[0-9]|3[0-1])\.|192\.168\.)', ip_str) else 0
            version = 4 if re.match(r'^\d+\.\d+\.\d+\.\d+$', ip_str) else (6 if ':' in ip_str else 0)
            return pd.Series({'IsPrivateIP': private, 'IPVersion': version})
        ip_features = df['LastLoginIP'].apply(extract_ip)
        df = pd.concat([df, ip_features], axis=1).drop(columns=['LastLoginIP'])
        print("✅ LastLoginIP parsé → IsPrivateIP + IPVersion")
    
    return df

def remove_useless_features(df: pd.DataFrame, missing_threshold: float = 0.5) -> pd.DataFrame:
    """5. Suppression features inutiles (corrigé pour les colonnes datetime/objet)"""
    df = df.copy()
    constant_cols = []
    for col in df.columns:
        # Variance nulle ou constante (nunique <= 1)
        if df[col].nunique(dropna=True) <= 1:
            constant_cols.append(col)
        # Var = 0 uniquement sur les colonnes numériques
        elif pd.api.types.is_numeric_dtype(df[col]):
            if df[col].var(ddof=0) == 0:
                constant_cols.append(col)
    
    if constant_cols:
        print(f"🗑️ Features constantes / variance nulle supprimées : {constant_cols}")
        df = df.drop(columns=constant_cols)
    
    # Trop de NaN
    high_missing = df.isnull().mean()[df.isnull().mean() > missing_threshold].index.tolist()
    if high_missing:
        print(f"🗑️ Features >{missing_threshold*100}% NaN supprimées : {high_missing}")
        df = df.drop(columns=high_missing)
    
    return df

def impute_missing_values(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.impute import KNNImputer, SimpleImputer
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if num_cols:
        knn = KNNImputer(n_neighbors=5)
        X_train[num_cols] = knn.fit_transform(X_train[num_cols])
        X_test[num_cols] = knn.transform(X_test[num_cols])
        print(f"✅ KNNImputer sur {len(num_cols)} features numériques")
    
    if cat_cols:
        mode = SimpleImputer(strategy='most_frequent')
        X_train[cat_cols] = mode.fit_transform(X_train[cat_cols])
        X_test[cat_cols] = mode.transform(X_test[cat_cols])
        print(f"✅ Mode Imputer sur {len(cat_cols)} features catégorielles")
    return X_train, X_test

def apply_pca(X_train: pd.DataFrame, X_test: pd.DataFrame, n_components=None, 
              variance_threshold: float = 0.95, save_model: str = "models/pca_model.pkl") -> Tuple[pd.DataFrame, pd.DataFrame, PCA]:
    os.makedirs(os.path.dirname(save_model), exist_ok=True)
    if n_components is None:
        pca_temp = PCA()
        pca_temp.fit(X_train)
        cum_var = np.cumsum(pca_temp.explained_variance_ratio_)
        n_components = int(np.argmax(cum_var >= variance_threshold) + 1)
        print(f"✅ ACP : {n_components} composantes ({variance_threshold*100:.0f}% variance)")
    
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    joblib.dump(pca, save_model)
    print(f"💾 PCA sauvegardé → {save_model}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(pca.explained_variance_ratio_)+1), np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.axhline(y=variance_threshold, color='r', linestyle='--')
    plt.title('Variance expliquée cumulée - ACP')
    plt.xlabel('Nombre de composantes')
    plt.ylabel('Variance cumulée')
    plt.grid(True)
    plt.savefig("reports/pca_explained_variance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    cols = [f'PC{i+1}' for i in range(n_components)]
    return pd.DataFrame(X_train_pca, columns=cols, index=X_train.index), \
           pd.DataFrame(X_test_pca, columns=cols, index=X_test.index), pca