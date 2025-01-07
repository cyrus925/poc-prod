# Stack Overflow Tag Predictor

Ce projet est un système de prédiction de tags pour les questions Stack Overflow. Il utilise le machine learning pour suggérer automatiquement des tags appropriés basés sur le contenu de la question.

## Structure du Projet

```
poc-to-prod-capstone/
├── predict/
│   ├── predict/
│   │   ├── run.py
│   │   └── app.py
│   └── tests/
│       └── test_predict.py
├── preprocessing/
│   └── preprocessing/
│       ├── embeddings.py
│       └── utils.py
└── train/
    ├── train/
    │   └── run.py
    ├── conf/
    │   └── train-conf.yml
    ├── data/
    │   └── stackoverflow_posts.csv
    └── tests/
        └── test_model_train.py
```

## Interface Utilisateur (app.py)

L'application web permet de prédire les tags d'une question Stack Overflow :
- Interface simple avec un champ de texte
- Prédiction instantanée des tags les plus pertinents
- Accessible via `http://localhost:8001`

### Installation et Lancement

```bash
# Installation des dépendances
pip install flask tensorflow transformers numpy

# Lancement de l'application
python -m predict.predict.app
```

## Pipeline de Prédiction (predict/run.py)

La classe `TextPredictionModel` gère les prédictions :
- Chargement du modèle entraîné
- Transformation du texte en embeddings BERT
- Prédiction des tags les plus probables

### Tests de Prédiction

Les tests (`test_predict.py`) vérifient :
- Le chargement correct du modèle
- La précision des prédictions
- La gestion des erreurs

```bash
python -m pytest predict/tests/test_predict.py
```

## Entraînement du Modèle (train/run.py)

Le processus d'entraînement comprend :
1. Chargement et prétraitement des données
2. Construction du modèle (architecture Dense)
3. Entraînement et évaluation
4. Sauvegarde des artefacts

### Configuration (train-conf.yml)

```yaml
batch_size: 32
epochs: 5
dense_dim: 64
min_samples_per_label: 10
verbose: 1
```

### Lancement de l'Entraînement

```bash
python -m train.train.run train/data/stackoverflow_posts.csv train/conf/train-conf.yml train/models
```

### Tests d'Entraînement

Les tests (`test_model_train.py`) vérifient :
- Le chargement correct des données
- L'entraînement du modèle
- La sauvegarde des artefacts



## Prétraitement (preprocessing/)

Le module de prétraitement gère :
- La transformation des textes en embeddings BERT
- Le nettoyage et la préparation des données
- La gestion des séquences d'entraînement


## Dépendances Principales

- TensorFlow 2.x
- Transformers (BERT)
- Flask
- NumPy
- Pytest

