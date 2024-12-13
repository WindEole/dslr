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
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from describe import (
    find_file,
    ft_count,
    ft_mean,
    ft_percentile,
    ft_std,
    load,
)
from scatter_plot import calculate_correlation


def save_results(
        models: list,
        features: list,
        class_names: list,
        file_path: str,
        train_data: pd.DataFrame,
        ) -> None:
    """Sauvegarde les coeff du modèle dans un fichier JSON.

    Les données seront organisées par classe, et incluent les min/max
    pour normalisation.
    Paramètres:
        model (LogisticRegression): Liste des modèles de régression logistique.
        features (list): Liste des noms des colonnes utilisées comme features.
        class_names (list): Liste des noms de classes.
        file_path (str): Chemin du fichier de sortie.
        train_data (pd.DataFrame): Données d'entraînement pour stats de norm.
    """
    coeff = {}
    for idx, class_name in enumerate(class_names):
        # Accéder à chaque modèle individuel dans la liste
        model = models[idx]  # Obtenir le modele sous forme de dico
        theta = model["theta"]  # Accède à theta
        bias = model["bias"]  # Accède à bias

        # Enregister les coefficients et intercepts pour chaque classe
        coeff[class_name] = {
            feature: theta[i] for i, feature in enumerate(features)
        }
        coeff[class_name]["intercept"] = bias

    # Calculer les stats de normalisation
    train_stat = train_data[features]
    # train_min = ft_min(train_stat)
    # train_max = ft_max(train_stat)
    train_mean = ft_mean(train_stat)
    train_std = ft_std(train_stat)

    # Créer le résultat final à sauvegarder
    result = {
        "coefficients": coeff,
        "normalisation_stats": {
            "mean": train_mean,
            "std": train_std,
        },
        "features": features,
    }

    # Sauvegarder les résultats dans un fichie JSON
    file_path = Path(file_path)
    with file_path.open("w") as file:
        json.dump(result, file, indent=4)

    print(f"Coefficients et statistiques sauvegardés dans '{file_path}'")


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Convertit la sortie linéaire en probabilité 0 à 1(fct sigmoïde)."""
    return 1 / (1 + np.exp(-z))


def compute_cost(h: np.ndarray, y: np.ndarray) -> np.float64:
    """Réalise la fonction de coût pour la régression logistique."""
    m = y.size
    cost = -(1 / m) * (np.dot(y, np.log(h)) + np.dot(1 - y, np.log(1 - h)))
    return cost


def gradient_descent(
        x: np.ndarray,
        y: np.ndarray,
        theta: float,
        bias: float,
        learning_rate: float,
        num_iter: int,
        ) -> tuple:
    """Effectue la descente de gradient pour ajuster les paramètres."""
    m = x.shape[0]
    for i in range(num_iter):
        # calcul du modèle et des prédictions
        model = np.dot(x, theta) + bias  # model est un np.ndarray
        predictions = sigmoid(model)

        # calcul des gradients
        error = predictions - y
        d_theta = np.dot(x.T, error) / m
        d_bias = np.sum(error) / m

        # mise à jour des paramètres
        theta -= learning_rate * d_theta
        bias -= learning_rate * d_bias

        # Calcul et affichage du coût
        if i % 100 == 0:
            cost = compute_cost(predictions, y)
            print(f"Iteration {i}: Cost {cost}")
    return theta, bias


def predict_proba_binary(x: pd.DataFrame, theta: any, bias: any) -> np.ndarray:
    """Prédit les probabilités pour un modèle binaire."""
    return sigmoid(np.dot(x, theta) + bias)


def predict_ova(x: np.ndarray, models: list, num_classes: int) -> np.ndarray:
    """Prédit les étiquettes pour un modèle One-vs-All."""
    predictions = np.zeros((x.shape[0], num_classes))
    for i in range(num_classes):
        model = models[i]
        theta = model["theta"]
        bias = model["bias"]
        predictions[:, i] = predict_proba_binary(x, theta, bias)
    return np.argmax(predictions, axis=1)


def logistic_regression_binary(
        x: np.ndarray,
        y: np.ndarray,
        learning_rate: float,
        num_iter: int,
        ) -> dict:
    """Entraîne un modèle de régression logistique binaire."""
    # Initialisation des paramètres
    theta = np.zeros(x.shape[1])
    bias = 0
    # Descente de gradient
    theta, bias = gradient_descent(x, y, theta, bias, learning_rate, num_iter)
    # On retourne un dico
    return {"theta": theta, "bias": bias}


def logistic_regression_ova(
        x: np.ndarray,
        y: np.ndarray,
        num_classes: int,
        learning_rate: float,
        num_iter: int,
        ) -> list:
    """Entraîne des modèles One-vs-All pour la régression logistique."""
    models = []
    for i in range(num_classes):
        print(f"Training model for class {i}...")
        binary_y = (y == i).astype(int)
        model = logistic_regression_binary(x, binary_y, learning_rate, num_iter)
        models.append(model)
    return models


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise les colonnes d'un DataFrame autour de l'axe 0."""
    df_mean = ft_mean(df)  # dico des moyennes
    df_std = ft_std(df)  # dico des écarts-types

    df_standardized = pd.DataFrame()
    for col in df.columns:
        if df_std[col]:  # Vérifie si le std n'est pas None ou zéro
            df_standardized[col] = (df[col] - df_mean[col]) / df_std[col]
        else:
            df_standardized[col] = df[col]  # Si std= None ou zéro, copie brute
    return df_standardized


def train_test_split(
        x: np.ndarray,
        y: np.ndarray,
        test_size: float,
        random_state: int,
        ) -> any:
    """Divise les données en ensembles d'entrainement et de test."""
    # Créer un générateur avec une graine
    rng = np.random.default_rng(seed=random_state)

    # Générer des indices mélangés
    indices = np.arange(x.shape[0])
    rng.shuffle(indices)

    # Division des données
    split_index = int(x.shape[0] * (1 - test_size))
    x_train = x[indices[:split_index]]
    y_train = y[indices[:split_index]]
    x_test = x[indices[split_index:]]
    y_test = y[indices[split_index:]]
    return x_train, x_test, y_train, y_test


def encoding_label(y: pd.Series) -> pd.Series:
    """Encode les étiquettes de maison par un int."""
    # Identifier les étiquettes uniques
    unique_labels = y.unique()

    # Donner un numero à chaque étiquette unique
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}

    # Encoder les étiquettes
    y_encoded = y.map(label_to_int)

    # print("Étiquettes uniques:", unique_labels)
    # print("Mapping (label -> int):", label_to_int)
    # print("Étiquettes encodées:", y_encoded.head())
    return y_encoded


def calculate_median(data: pd.DataFrame) -> dict:
    """Calculate median (equal to percentile 50)."""
    # On sélectionne uniquement les colonnes numériques
    num_data = data.select_dtypes(include=["number"])
    if num_data.empty:
        print("Aucune colonne numérique dans les données.")
        return None

    # COUNT = nb d'observations non nulles
    count_values = ft_count(num_data)
    median = {}
    for col in num_data.columns:
        # Exclure les valeurs nulles pour le calcul du min
        non_null_values = num_data[~num_data[col].isna()][col]
        median[col] = ft_percentile(50, non_null_values, count_values[col])

    return median


def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """Remplace toutes les valeurs manquantes.

    On détermine l'a/symétrie des données avec l'indicateur skewness.
    Si les données sont sym (skewness ~ 0): on remplace les Nan par la moyenne.
    Si elles sont asym: on remplace par la médiane (=percentile 50).
    """
    mean_val = ft_mean(data.select_dtypes(include=["number"]).drop(
        ["Index"], axis=1, errors="ignore",
        ))
    median_val = calculate_median(data)
    # Traitement de l'a/symétrie
    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            skewness = data[column].skew()

            # Remplacement des valeurs manquantes selon l'asymétrie
            if -0.5 <= skewness <= 0.5:
                for col, mean_value in mean_val.items():
                    if mean_value is not None:
                        data.fillna({col: mean_value}, inplace=True)
            else:
                for col, median_value in median_val.items():
                    if median_value is not None:
                        data.fillna({col: median_value}, inplace=True)
    return data


def handle_missing_correlated_values(data: pd.DataFrame) -> pd.DataFrame:
    """Remplace les valeurs manquantes dans les matières corrélées.

    On détermine le facteur de correlation, puis on remplace les Nan par les
    valeurs calculées avec ce facteur. Si les valeurs manquent dans les deux
    matières, on laisse Nan.
    """
    # Traitement des correlations:
    high_corr = calculate_correlation(data, 0.9)
    if high_corr.empty:
        print("Aucune paire de variables avec corrélation élevée.")
        return data

    col_a = high_corr.iloc[0]["Variable 1"]
    col_b = high_corr.iloc[0]["Variable 2"]
    # print(f"{col_a} || {col_b}")
    # missing_data_rows = data[data[[col_a, col_b]].isnull().any(axis=1)]
    # print("Lignes avec des valeurs manquantes: ")
    # print(missing_data_rows[[col_a, col_b]])
    data_cleaned = data.dropna(subset=[col_a, col_b])
    # Determination du facteur de multiplication:
    x = data_cleaned[col_a]
    y = data_cleaned[col_b]
    slope, intercept = np.polyfit(x, y, 1)
    print(f"Facteur multiplicateur: {slope}")
    # impute col_a using col_b
    data[col_a] = data.apply(
        lambda row: row[col_b] / slope if pd.isna(row[col_a]) and not pd.isna(row[col_b]) else row[col_a],
        axis=1,
    )
    # impute col_b using col_a
    data[col_b] = data.apply(
        lambda row: row[col_a] * slope if pd.isna(row[col_b]) and not pd.isna(row[col_a]) else row[col_b],
        axis=1,
    )
    # print(data[[col_a, col_b]])
    return data


def logistic_regression(data: pd.DataFrame) -> None:
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
        msg = "Aucune colonne 'House' n'a éte trouvée"
        raise ValueError(msg)

    # Compléter les valeurs Nan
    data = handle_missing_correlated_values(data)
    data = handle_missing_values(data)

    # Sélection des matières et de la cible
    # subjects = ["Herbology", "Defense Against the Dark Arts", "Astronomy", "Ancient Runes"]
    subjects = [
        col for col in data.columns
        if pd.api.types.is_numeric_dtype(data[col]) and col.lower() != "index"
    ]
    if not all(subject in data.columns for subject in subjects + [house_col]):
        msg = "Les colonnes nécessaires sont absentes."
        raise ValueError(msg)

    # ----- PREPARATION DES DONNEES POUR LA REGRESSION ----- #

    # On sépare la colonne des Maison et le tableau des notes
    x_df = data[subjects]  # -> FEATURES (DataFrame)
    y_ser = data[house_col]  # -> LABELS (Series)

    # Encodage des étiquettes de y : chq Maison est représentée par un numero
    y = encoding_label(y_ser)

    # On passe les données à numpy (calculs plus rapides)
    x = x_df.to_numpy()
    y = y.to_numpy()

    # Division des données en ensembles d'entraînement et de test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Standardiser les données sur np reconvertis en DataFrame (compatibilité)
    x_train = standardize(pd.DataFrame(x_train, columns=subjects))
    x_test = standardize(pd.DataFrame(x_test, columns=subjects))

    # Régression logistique One-vs-All
    num_classes = len(y_ser.unique())
    models = logistic_regression_ova(x_train, y_train, num_classes, learning_rate=0.01, num_iter=1000)

    # Évaluation
    y_pred = predict_ova(x_test.to_numpy(), models, num_classes)
    print("\nRapport de classification :")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.2f}")

    # Sauvegarde des coefficients
    features = x_df.columns.tolist()
    class_names = y_ser.unique().tolist()
    save_results(models, features, class_names, "logreg_coeff.json", x_df)


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
        print(f"Erreur : le fichier '{filename}' n'a pas été trouvé.")
        sys.exit(1)

    print(f"Fichier {filename} trouvé : {file_path}")
    data = load(file_path)
    if data is None:
        sys.exit(1)
    try:
        logistic_regression(data)
    except KeyboardInterrupt:
        print("\nInterruption du programme par l'utilisateur (Ctrl + C)")
        sys.exit(0)  # Sort proprement du programme
    except ValueError as e:
        print(e)


if __name__ == "__main__":
    main()
