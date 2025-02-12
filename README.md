# Projet : Machine Learning Pipeline

Ce dépôt contient trois scripts Python dédiés aux tâches de prétraitement, de classification et de clustering dans un workflow de Machine Learning.

## Contenu du dépôt

1. **classificationPipe.py** : Script interactif pour la classification de données, incluant la sélection de modèle, l'entraînement et l'évaluation.
2. **featureEngineering.py** : Script interactif permettant d'effectuer de l'ingénierie des caractéristiques sur un jeu de données.
3. **kmeans.py** : Implémentation de la méthode Elbow, de l'analyse des silhouettes et de la statistique du Gap pour déterminer le nombre optimal de clusters en K-Means.

---

## 1. classificationPipe.py

### Description
Ce script guide l'utilisateur à travers les étapes suivantes :
- Chargement et prétraitement des données
- Gestion des valeurs manquantes
- Encodage des variables catégorielles
- Sélection et entraînement de plusieurs modèles de classification (RandomForest, SVM, KNN, etc.)
- Évaluation et comparaison des performances des modèles

### Dépendances
- `pandas`
- `scikit-learn`
- `tqdm`

### Utilisation
```sh
python classificationPipe.py
```
L'utilisateur devra fournir un fichier CSV et choisir la colonne cible.

---

## 2. featureEngineering.py

### Description
Permet d'effectuer les tâches suivantes :
- Chargement des données
- Analyse des relations entre les variables (optionnel)
- Gestion des valeurs manquantes (imputation ou suppression)
- Création de nouvelles features à partir d'expressions personnalisées

### Dépendances
- `pandas`
- `matplotlib`
- `seaborn`

### Utilisation
```sh
python featureEngineering.py
```
L'utilisateur devra fournir un fichier CSV et suivre les instructions interactives.

---

## 3. kmeans.py

### Description
Ce script aide à déterminer le nombre optimal de clusters pour un modèle K-Means en utilisant :
- La méthode Elbow
- L'analyse des silhouettes
- La statistique du Gap

### Dépendances
- `numpy`
- `scikit-learn`
- `matplotlib`
- `gap-statistic`

### Utilisation
```sh
python kmeans.py
```
Affiche les graphes pour aider à choisir la meilleure valeur de `k`.

---

## Installation des dépendances
Si ce n'est pas déjà fait, installez les bibliothèques requises avec :
```sh
pip install pandas scikit-learn tqdm matplotlib seaborn gap-statistic
```

---

## Auteur
Développé par ThomasLyy.

