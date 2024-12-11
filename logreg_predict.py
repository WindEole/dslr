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


def handle_missing_values(data: pd.DataFrame, mean_val: dict):
    """Remplace les valeurs manquantes dans le DataFrame.

    Si matières = Defense Against Dark Arts ou Astronomy : correlation * -100.
    pour les autres matières: on remplace par la moyenne.
    """
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
    for col, mean_val in mean_val.items():
        if mean_val is not None:
            data.fillna({col: mean_val}, inplace=True)
    return data


def main() -> None:
    """Load data and visualize."""
    # required_subjects = ["Herbology", "Defense Against the Dark Arts", "Astronomy", "Ancient Runes"]
    # required_subjects = ["Divination", "History of Magic", "Charms", "Muggle Studies"]
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
            # Calculer les moyennes des données d'entrainement
            mean_val = ft_mean(data)
            # Remplacer les valeurs manquantes dans le jeu de données
            data_filled = handle_missing_values(data, mean_val)
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
    else:
        print(f"Erreur : le fichier '{filename}' n'a pas été trouvé.")


if __name__ == "__main__":
    main()
