# 🛍️ Analyse Comportementale Clientèle Retail
### Atelier Machine Learning — GI2 | 2025-2026

> Pipeline complet de Data Science : Exploration → Préparation → Modélisation → Évaluation → Déploiement Flask

---

## 📋 Description

Ce projet implémente une chaîne complète de traitement Machine Learning sur une base de données e-commerce de **4 372 clients** avec **52 features** comportementales.

**Objectifs :**
- Segmenter les clients (clustering KMeans)
- Prédire le risque de churn (RandomForest)
- Déployer une interface web interactive (Flask)

---

## 🗂️ Structure du projet

```
projet_ml_retail/
├── data/
│   ├── raw/               # retail.csv (données brutes)
│   ├── processed/         # données nettoyées + PCA
│   └── train_test/        # X_train, X_test, y_train, y_test
├── notebooks/             # prototypage Jupyter
├── src/
│   ├── preprocessing.py   # pipeline complet (nettoyage → split → scale)
│   ├── train_model.py     # clustering + classification
│   ├── predict.py         # prédiction sur nouveaux clients
│   └── utils.py           # fonctions utilitaires
├── models/                # modèles sauvegardés (.pkl)
├── app/
│   ├── app.py             # application Flask
│   └── templates/         # HTML (base, index, predict, dashboard)
├── reports/               # visualisations générées
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

### 1. Cloner le dépôt
```bash
git clone https://github.com/VOTRE_USERNAME/projet_ml_retail.git
cd projet_ml_retail
```

### 2. Créer et activer l'environnement virtuel
```bash
# Créer
python -m venv venv

# Activer (Windows)
venv\Scripts\activate

# Activer (Mac/Linux)
source venv/bin/activate
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

---

## 🚀 Guide d'utilisation

### Étape 1 — Préparation des données
```bash
python -m src.preprocessing
```
> Génère les fichiers dans `data/train_test/` et `data/processed/`

### Étape 2 — Entraînement des modèles
```bash
python -m src.train_model
```
> Entraîne KMeans + RandomForest, sauvegarde dans `models/`
> Génère la matrice de confusion dans `reports/`

### Étape 3 — Prédiction batch (optionnel)
```bash
python -m src.predict
```
> Prédit le churn sur le jeu de test, résultats dans `reports/predictions.csv`

### Étape 4 — Lancer l'application Flask
```bash
python app/app.py
```
> Ouvrir [http://127.0.0.1:5000](http://127.0.0.1:5000) dans le navigateur

---

## 📊 Résultats

| Métrique   | Score  |
|------------|--------|
| Accuracy   | ~96%   |
| F1-Score   | ~94%   |
| Precision  | ~96%   |
| Recall     | ~92%   |

> Note : Scores élevés car dataset synthétique avec patterns clairs.

### Segmentation KMeans (4 clusters)
| Segment     | Description                        |
|-------------|------------------------------------|
| Champions   | Haute fréquence, haute valeur      |
| Fidèles     | Réguliers, bon potentiel           |
| Potentiels  | Actifs, pas encore réguliers       |
| Dormants    | Inactifs, risque de churn élevé    |

---

## 🛡️ Traitement du Data Leakage

Colonnes supprimées pour éviter la fuite d'information vers la target `Churn` :

- `ChurnRisk`, `AccountStatus`, `CustomerType` — dérivées de Churn
- `RFMSegment`, `LoyaltyLevel`, `SpendingCategory` — calculées post-churn
- `SatisfactionScore`, `SupportTicketsCount` — enregistrées après l'événement
- `Recency`, `FavoriteSeason`, `CustomerTenureDays`, `PreferredMonth` — corrélation > 0.4

---

## 🌐 Interface Flask

| Route         | Description                              |
|---------------|------------------------------------------|
| `/`           | Page d'accueil (présentation projet)     |
| `/predict`    | Formulaire de prédiction individuelle    |
| `/dashboard`  | Dashboard métriques du modèle            |
| `/api/predict`| API REST JSON (POST)                     |

### Exemple API REST
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"Frequency": 3, "MonetaryTotal": 150, "UniqueProducts": 10}'
```

---

## 📦 Dépendances principales

```
pandas >= 2.0
numpy >= 1.24
scikit-learn >= 1.3
flask >= 3.0
joblib >= 1.3
matplotlib >= 3.7
seaborn >= 0.12
```

---

## 👤 Auteur

Projet réalisé dans le cadre du **Module Machine Learning — GI2**  
Encadrant : Fadoua Drira | Année Universitaire 2025-2026