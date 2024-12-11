"""Logistic regression program.

A partir du pair plot vu précedemment, on repère les paires de matières
pour lesquelles les clusters d'élèves sont les plus distincts :
- Herbology / Astronomy
- Defense Against The Dark Arts / Herbology
- Ancient Runes / Astronomy
- Ancient Runes / Herbology
- Ancient Runes / Defense Against The Dark Arts

Ce qui se réduit à 4 matières pertinentes:
Herbology / Astronomy / Ancient Runes / Defense Against The Dark Arts
"""
import json
import re
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from describe import (
    extract_tgz,
    find_file,
    ft_max,
    ft_mean,
    ft_min,
    load,
)
from histogram import normalize_data


def save_results(
        model,
        features: list,
        class_names: list,
        file_path: str,
        train_data: pd.DataFrame,
        ) -> None:
    """Sauvegarde les coeff du modèle dans un fichier JSON.

    Les données seront organisées par classe, et incluent les min/max
    pour normalisation.
    Paramètres:
        model (LogisticRegression): Modèle de régression logistique.
        features (list): Liste des noms des colonnes utilisées comme features.
        class_names (list): Liste des noms de classes.
        file_path (str): Chemin du fichier de sortie.
        train_data (pd.DataFrame): Données d'entraînement pour stats de norm.
    """
    coeff = {}
    for idx, class_name in enumerate(class_names):
        coeff[class_name] = {
            feature: model.coef_[idx][i] for i, feature in enumerate(features)
        }
        coeff[class_name]["intercept"] = model.intercept_[idx]

    train_stat = train_data[features]
    train_min = ft_min(train_stat)
    train_max = ft_max(train_stat)

    result = {
        "coefficients": coeff,
        "normalisation_stats": {
            "min": train_min,
            "max": train_max,
        },
        "features": features,
    }

    with open(file_path, "w") as file:
        json.dump(result, file, indent=4)

    print(f"Coefficients et statistiques sauvegardés dans '{file_path}'")


def sigmoid(z):
    """Convertir la sortie linéaire en probabilité 0 à 1."""
    return 1 / ( 1 + np.exp(-z))


def compute_cost(X, y, beta):
    m = len(y)
    z = np.dot(X, beta)
    h = sigmoid(z)
    cost = -(1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost


def gradient_descent(X, y, beta, learning_rate, iterations):
    m = len(y)
    for _ in range(iterations):
        z = np.dot(X, beta)
        h = sigmoid(z)
        gradient = (1 / m) * np.dot(X.T, (h - y))
        beta -= learning_rate * gradient
    return beta


def predict(X, beta, threshold=0.5):
    probabilities = sigmoid(np.dot(X, beta))
    return (probabilities >= threshold).astype(int)


def logistic_regression(data: pd.DataFrame):
    """Fonction de régression logistique de type one-vs-all.

    Effectue une régression logistique One-vs-All pour classer les étudiants
    en fonction de leurs scores dans 4 matières principales.
    """
    # On cherche une colonne contenant House, avec des variantes
    house_col = None
    for col in data.columns:
        if re.search(r"\b(House|Maison|Houses)\b", col, re.IGNORECASE):
            house_col = col
            break

    # Si aucune colonne "House" n'est trouvée, on lève une erreur
    if house_col is None:
        raise ValueError("Aucune colonne 'House' n'a éte trouvée")

    # Sélection des matières et de la cible
    subjects = ["Herbology", "Defense Against the Dark Arts", "Astronomy", "Ancient Runes"]
    # subjects = ["Herbology", "Defense Against the Dark Arts", "Astronomy", "Ancient Runes", "Divination", "History of Magic", "Charms", "Muggle Studies", "Arithmancy", "Transfiguration", "Potions", "Care of Magical Creatures", "Flying"]
    if not all(subject in data.columns for subject in subjects + [house_col]):
        raise ValueError("Les colonnes nécessaires (Herbology, Defense Against the Dark Arts, Astronomy, Ancient Runes, House) sont absentes.")

    X = data[subjects]
    y = data[house_col]

    # Normalisation des données pour une meilleure convergence
    X_scaled = normalize_data(X)

    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Régression logistique One-vs-All
    model = LogisticRegression(multi_class="ovr", max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Évaluation
    y_pred = model.predict(X_test)
    print("Rapport de classification :")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.2f}")

    # Sauvegarde des coefficients
    features = X.columns.tolist()
    class_names = model.classes_.tolist()
    save_results(model, features, class_names, "logreg_coeff.json", X)


def handle_missing_values(data: pd.DataFrame, mean_val: dict) -> pd.DataFrame:
    """Remplace les valeurs manquantes des col Defense et Astronomy."""
    missing_data_rows = data[data[["Defense Against the Dark Arts", "Astronomy"]].isnull().any(axis=1)]
    print("Lignes avec des valeurs manquantes: ")
    print(missing_data_rows[["Defense Against the Dark Arts", "Astronomy"]])
    col_a = "Defense Against the Dark Arts"  # = / -100 Astronomy
    col_b = "Astronomy"  # = * -100 Defense
    # impute col_a using col_b
    data[col_a] = data.apply(
        lambda row: -row[col_b] / 100 if pd.isna(row[col_a]) and not pd.isna(row[col_b]) else row[col_a],
        axis=1,
    )
    # impute col_b using col_a
    data[col_b] = data.apply(
        lambda row: -row[col_a] * 100 if pd.isna(row[col_b]) and not pd.isna(row[col_a]) else row[col_b],
        axis=1,
    )
    print(data)
    for col, mean_val in mean_val.items():
        if mean_val is not None:
            data.fillna({col: mean_val}, inplace=True)
    print(data)
    return data


def main() -> None:
    """Load data and visualize."""
    # On vérifie si l'arg en ligne de commande est fourni
    if len(sys.argv) != 2:
        print("Usage: python logreg_train.py fichier.csv")
        sys.exit(1)

    filename = sys.argv[1]
    dir_path = "./"
    file_path = find_file(filename, dir_path)

    if not file_path:
        tgz_file = find_file("datasets.tgz", dir_path)
        if tgz_file:
            print(f"Fichier {tgz_file} trouvé. Décompression en cours...")
            extract_tgz(tgz_file, dir_path)
            # rechercher à nouveau le fichier.csv
            file_path = find_file(filename, dir_path)
        else:
            print(f"Erreur : fichier '{filename}' et fichier.tgz absents.")
            sys.exit(1)

    if file_path:
        print(f"Fichier {filename} trouvé : {file_path}")
        data = load(file_path)
        if data is None:
            sys.exit(1)
        print(data)
        try:
            mean_val = ft_mean(data.select_dtypes(include=["number"]).drop(["Index"], axis=1, errors="ignore"))
            data_filled = handle_missing_values(data, mean_val)
            data_cleaned = data_filled.dropna()
            logistic_regression(data_cleaned)
        except KeyboardInterrupt:
            print("\nInterruption du programme par l'utilisateur (Ctrl + C)")
            sys.exit(0)  # Sort proprement du programme
        except ValueError as e:
            print(e)
    else:
        print(f"Erreur : le fichier '{filename}' n'a pas été trouvé.")


if __name__ == "__main__":
    main()
