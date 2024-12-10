"""Programme de prediction.

Ce programme récupère les coefficients établis par le programme d'entrainement
par regression logistique, et les applique à un nouveau jeu de données pour
prédire l'appartenance de chaque élève à telle ou telle maison.
"""
import sys
import json

import numpy as np
import pandas as pd

from describe import (
    extract_tgz,
    find_file,
    ft_max,
    ft_mean,
    load,
)


def predict_class(data: pd.DataFrame, coefficients: dict) -> pd.Series:
    predictions = []
    for _, row in data.iterrows():
        class_scores = {}
        for class_name, coeffs in coefficients.items():
            intercept = coeffs["intercept"]
            score = intercept + sum(row[feature] * coeff for feature, coeff in coeffs.items() if feature != "intercept")

            # Clipping du score pour éviter les overflow
            score = np.clip(score, -500, 500)

            # Calculer la probabilité avec la fonction sigmoïde
            probability = 1 / (1 + np.exp(-score))
            class_scores[class_name] = probability
        # print(f"Scores: {class_scores}")
        # Prédire la classe avec la probabilité maximale
        # max_val = ft_max(class_scores, key=class_scores.get)
        # print(f"max val = {max_val}")
        # key = class_scores.get
        # print(f"key is {key}")
        predictions.append(ft_max(class_scores))
        # predicted_class = ft_max(class_scores, key=class_scores.get)
        # print(f"Predicted class: {predicted_class}")
    return pd.Series(predictions)


def normalize(data: pd.DataFrame, train_min: pd.Series, train_max: pd.Series) -> pd.Series:
    """Normalize the grades of all subjects between 0.0 and 1.0."""
    subjects = data.select_dtypes(include=["number"]).drop(
        ["Index"],
        axis=1,
        errors="ignore")
    # print(subjects)
    # print(f"{train_min}\n{train_max}")
    subjects_norm = (subjects - train_min) / (train_max - train_min)
    return subjects_norm


def handle_missing_values(data: pd.DataFrame, mean_val: dict):
    """Remplacer les valeurs manquantes dans le DataFrame avec les moyennes spécifiées."""
    for col, mean_val in mean_val.items():
        if mean_val is not None:
            data.fillna({col: mean_val}, inplace=True)  # Remplacer NaN par la moyenne dans chaque colonne
    return data


def main() -> None:
    """Load data and visualize."""
    required_subjects = ["Herbology", "Defense Against the Dark Arts", "Astronomy", "Ancient Runes"]
    # On vérifie si l'arg en ligne de commande est fourni
    if len(sys.argv) != 2:
        print("Usage: python logreg_predict.py fichier.csv")
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
        try:
            # Vérification des colonnes nécessaires
            if not all(subject in data.columns for subject in required_subjects):
                raise ValueError(f"Le jeu de données doit contenir les colonnes suivantes : {required_subjects}")
            # Sélection des colonnes nécessaires uniquement
            data = data[required_subjects]
            # Récupération des info depuis le fichier JSON
            with open("logreg_coeff.json", "r") as file:
                coefficients_data = json.load(file)
            coefficients = coefficients_data["coefficients"]
            norm_stats = coefficients_data["normalisation_stats"]
            train_min = pd.Series(norm_stats["min"])
            train_max = pd.Series(norm_stats["max"])
            # Calculer les moyennes des données d'entrainement
            mean_val = ft_mean(data)
            # Remplacer les valeurs manquantes dans le jeu de données
            data_filled = handle_missing_values(data, mean_val)
            # Normalisation ATTENTION : avec min et max du jeu d'entrainement !
            data_norm = normalize(data_filled, train_min, train_max)
            # print(f"data normalisées : \n{data_norm}")  # OK BON
            # Prédire les classes
            predicted_classes = predict_class(data_norm, coefficients)
            print(predicted_classes)
        except KeyboardInterrupt:
            print("\nInterruption du programme par l'utilisateur (Ctrl + C)")
            sys.exit(0)  # Sort proprement du programme
        except ValueError as e:
            print(e)
    else:
        print(f"Erreur : le fichier '{filename}' n'a pas été trouvé.")


if __name__ == "__main__":
    main()