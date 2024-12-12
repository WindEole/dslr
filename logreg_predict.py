"""Programme de prediction.

Ce programme récupère les coefficients établis par le programme d'entrainement
par regression logistique, et les applique à un nouveau jeu de données pour
prédire l'appartenance de chaque élève à telle ou telle maison.
"""
import json
import sys

import numpy as np
import pandas as pd

from describe import (
    find_file,
    ft_max,
    ft_mean,
    load,
)
from logreg_train import (
    calculate_median,
    handle_missing_correlated_values,
    handle_missing_values,
)


def predict_class(data: pd.DataFrame, coefficients: dict) -> pd.Series:
    """Predit la maison où doit être réparti chaque élève.

    param:
        data: DataFrame des données (Nan remplacées + normalisées)
        coefficients: dict des poids calculés par le training program.
    """
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

        # Prédire la classe avec la probabilité maximale
        predictions.append(ft_max(class_scores))
    return pd.Series(predictions)


def normalize(
        data: pd.DataFrame, train_min: pd.Series, train_max: pd.Series,
        ) -> pd.Series:
    """Normalize the grades of all subjects between 0.0 and 1.0."""
    subjects = data.select_dtypes(include=["number"]).drop(
        ["Index"],
        axis=1,
        errors="ignore")
    subjects_norm = (subjects - train_min) / (train_max - train_min)
    return subjects_norm


def replace_nan_val(data: pd.DataFrame) -> pd.DataFrame:
    """Fonction qui gère le remplacement des valeurs Nan."""
    data = handle_missing_correlated_values(data)
    mean_val = ft_mean(data.select_dtypes(include=["number"]).drop(["Index"], axis=1, errors="ignore"))
    median_val = calculate_median(data)
    data = handle_missing_values(data, mean_val, median_val)
    return data


def main() -> None:
    """Load data and visualize."""
    # On vérifie si l'arg en ligne de commande est fourni
    if len(sys.argv) != 2:
        print("Usage: python logreg_predict.py fichier.csv")
        sys.exit(1)

    filename = sys.argv[1]
    dir_path = "./"
    file_path = find_file(filename, dir_path)

    if not file_path:
        print(f"Erreur : le fichier '{filename}' n'a pas été trouvé.")
        sys.exit(1)

    print(f"Fichier {filename} trouvé : {file_path}")
    data = load(file_path)
    if data is None:
        sys.exit(1)
    try:
        # Récupération des info depuis le fichier JSON
        with open("logreg_coeff.json", "r") as file:
            coefficients_data = json.load(file)
        coefficients = coefficients_data["coefficients"]
        norm_stats = coefficients_data["normalisation_stats"]
        train_min = pd.Series(norm_stats["min"])
        train_max = pd.Series(norm_stats["max"])
        required_subjects = coefficients_data["features"]

        # Vérification des colonnes nécessaires dans les données à tester
        if not all(subject in data.columns for subject in required_subjects):
            raise ValueError(f"Le jeu de données doit contenir les colonnes suivantes : {required_subjects}")

        # Sélection des colonnes nécessaires uniquement
        data = data[required_subjects]
        data_filled = replace_nan_val(data)

        # Normalisation ATTENTION : avec min et max du jeu d'entrainement !
        data_norm = normalize(data_filled, train_min, train_max)
        # print(f"data normalisées : \n{data_norm}")  # OK BON

        # Prédire les classes
        predicted_classes = predict_class(data_norm, coefficients)

        # enregistrement dans un dataFrame et export pour visualisation
        houses_df = pd.DataFrame(predicted_classes)
        print(houses_df)
        houses_df.columns = ['Hogwarts House']  # on n'a qu'une col...
        houses_df['Index'] = houses_df.index  # création d'une col index
        houses_df = houses_df[['Index', 'Hogwarts House']]  # déplace index
        houses_df.to_csv("houses.csv", index=False)  # retire index default
        print("La répartition a été sauvegardée dans 'houses.csv'")

    except KeyboardInterrupt:
        print("\nInterruption du programme par l'utilisateur (Ctrl + C)")
        sys.exit(0)  # Sort proprement du programme
    except ValueError as e:
        print(e)


if __name__ == "__main__":
    main()
