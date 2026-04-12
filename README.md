# 🛍️ Analyse Comportementale Clientèle - Machine Learning

## 📌 Description du projet

Ce projet s’inscrit dans le cadre du module Machine Learning.

L’objectif est d’analyser le comportement des clients d’un e-commerce de cadeaux afin de :

* Comprendre les profils clients
* Identifier les facteurs influençant le churn
* Améliorer les stratégies marketing

---

##  Instructions d’installation

### 1. Cloner le projet

```bash
git clone <url_du_repo>
cd projet_ml_retail
```

### 2. Créer un environnement virtuel

```bash
python -m venv venv
```

### 3. Activer l’environnement

#### Windows :

```bash
venv\Scripts\activate
```

### 4. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

##  Structure du projet

```
projet_ml_retail/
│
├── data/
│   ├── raw/            # Données brutes
│   ├── processed/      # Données nettoyées
│   └── train_test/     # Données séparées
│
├── notebooks/          # Analyse exploratoire (Jupyter)
├── src/                # Scripts Python
├── models/             # Modèles sauvegardés
├── app/                # Application Flask
├── reports/            # Graphiques et résultats
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Guide d’utilisation

### Lancer l’analyse exploratoire

```bash
jupyter notebook
```

Puis ouvrir :

```
notebooks/01_exploration.ipynb
```

---

##  Remarques

* Les données nécessitent un nettoyage important (valeurs manquantes, incohérences)
* Le projet suit une pipeline complète :
  **Exploration → Préparation → Modélisation → Évaluation → Déploiement**


---

##  Contexte & Mission

Dans ce projet, nous jouons le rôle d’un **data scientist** au sein d’une entreprise e-commerce spécialisée dans la vente de cadeaux.

L’entreprise souhaite mieux comprendre sa clientèle afin de :

* Personnaliser ses stratégies marketing
* Réduire le taux de départ des clients (churn)
* Optimiser son chiffre d’affaires

Le dataset utilisé contient de nombreuses variables issues de données réelles.
Ces données sont volontairement imparfaites afin de permettre une maîtrise complète du processus de data science.

---

##  Objectifs pédagogiques

Ce projet couvre toutes les étapes d’un pipeline de Machine Learning :

| Compétence     | Description                                                     |
| -------------- | --------------------------------------------------------------- |
| Exploration    | Analyse de la qualité et de la structure des données            |
| Préparation    | Nettoyage, encodage et normalisation des données                |
| Transformation | Réduction de dimension (ACP)                                    |
| Modélisation   | Application de modèles (classification, clustering, régression) |
| Évaluation     | Interprétation des résultats                                    |
| Déploiement    | Création d’une application avec Flask                           |

---


---

##  Synthèse des features

Le dataset est composé de **plus de 50 variables** décrivant le comportement des clients.

###  Features numériques

Ces variables représentent des mesures quantitatives :

* Activité client : `Recency`, `Frequency`, `TotalTrans`
* Dépenses : `MonetaryTotal`, `MonetaryAvg`, `MonetaryMax`
* Quantités : `TotalQuantity`, `AvgQtyPerTrans`
* Temps : `CustomerTenure`, `FirstPurchase`
* Profil client : `Age`, `SupportTickets`, `Satisfaction`

---

###  Features catégorielles

Ces variables décrivent des catégories ou profils :

* Segmentation client : `RFMSegment`, `CustomerType`
* Comportement : `SpendingCat`, `BasketSize`, `ProdDiversity`
* Données démographiques : `Gender`, `Region`, `Country`
* Préférences : `PreferredTime`, `FavoriteSeason`
* Statut : `AccountStatus`, `LoyaltyLevel`

---

###  Variable cible

* `Churn` : indique si un client a quitté l’entreprise

  * 0 → client fidèle
  * 1 → client perdu

---

##  Problèmes de qualité des données

Le dataset contient plusieurs problèmes typiques du monde réel :

* ❗ Valeurs manquantes (ex : Age ~30%)
* ❗ Valeurs aberrantes (quantités négatives, valeurs extrêmes)
* ❗ Formats incohérents (dates)
* ❗ Variables inutiles (ex : Newsletter constante)
* ❗ Données complexes (IP à transformer)
* ❗ Déséquilibre des classes (Churn)

---

##  Objectif du preprocessing

Les données doivent être transformées pour :

* Nettoyer les erreurs
* Gérer les valeurs manquantes
* Encoder les variables catégorielles
* Normaliser les données
* Créer de nouvelles features pertinentes

---
