"""Scatter plot program.

Ce programme ouvre un fichier de données compressées au format .tgz et
génère une visualisation des données sous forme de nuage de points.
"""
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from describe import (
    find_file,
    ft_mean,
    ft_std,
    load,
)
from histogram import normalize_data


def close_on_enter(event: any) -> None:
    """Close the figure when the Enter key is pressed."""
    if event.key == "enter":  # Si la touche 'Enter' est pressée
        plt.close(event.canvas.figure)  # Ferme la figure associée


def extract_high_corr(
        corr_matrix: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Extract pairs of columns whose correlation is near 1 or -1.

    Argument:
        corr_matrix(pd.DataFrame): matrice de correlation.
        threshold (float): seuil de correlation absolue (0.9 par défaut).

    Retourne pd.DataFrame listant les paires de colonnes et leur corrélation.
    """
    # Convertir la matrice en long (simplifie filtrage, manip et présentation)
    corr_long = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool),
        )
    corr_long = corr_long.stack().reset_index()
    corr_long.columns = ["Variable 1", "Variable 2", "Correlation"]

    # Filtrer les paires selon le seuil
    high_corr = corr_long[(corr_long["Correlation"].abs() >= threshold)]
    return high_corr


def ft_corr(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate a correlation matrix.

    Reproduit la méthode corr() de pandas.
    arg: DataFrame avec des colonnes numériques centrées et réduites
    returns: pd.DataFrame -> matrice de correlation entre les col de data
    """
    data_mean = ft_mean(data)
    data_std = ft_std(data)
    data_cr = (data - data_mean) / data_std
    n = len(data)
    corr_matrix = data_cr.T.dot(data_cr) / n
    # Conversion en DataFrame pour avoir des noms de colonnes
    corr_matrix = pd.DataFrame(
        corr_matrix,
        index=data_cr.columns,
        columns=data_cr.columns,
        )
    return corr_matrix


def calculate_correlation(data: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Calculate correlation between subjects et returns the best.

    Param:
        data: original data (raw)
        threshold: seuil pour déterminer les meilleures correlations
    Return: un DataFrame des matières les plus correlées.
    """
    # Supprimer les lignes contenant des NaN
    data_cleaned = data.dropna()
    # Normalisation des données
    data_norm = normalize_data(data_cleaned)
    # Sélectionner les colonnes de données numériques (drop index)
    features = data_norm.select_dtypes(
        include=["number"]).drop(columns=["Index"],
                                 errors="ignore")
    # Reproduction de la méthode .corr()
    corr_matrix = ft_corr(features)
    # print(f"\nmethode corr: \n{features.corr()}")
    # print(f"\nmethode corr custom: \n{corr_matrix}")
    high_corr = extract_high_corr(corr_matrix, threshold)
    # print(f"\nHigh correlation pair :\n{high_corr}")
    return high_corr


def viz_scatterplot(data: pd.DataFrame) -> None:
    """Représente les données sous forme d'histogramme."""
    # On cherche une colonne contenant House, avec des variantes
    house_col = None
    for col in data.columns:
        if re.search(r"\b(House|Maison|Houses)\b", col, re.IGNORECASE):
            house_col = col
            break

    # Si aucune colonne "House" n'est trouvée, on lève une erreur
    if house_col is None:
        raise ValueError("Aucune colonne 'House' n'a éte trouvée")

    print(house_col)

    house_counts = data[house_col].value_counts()
    print("\nEffectifs par maison :")
    for maison, count in house_counts.items():
        print(f"{maison}: {count}")

    # Supprimer les lignes contenant des NaN
    data_cleaned = data.dropna()
    # Normalisation des données
    subjects_norm = normalize_data(data_cleaned)

    # # Sélectionner les colonnes de données numériques (drop index)
    # features = subjects_norm.select_dtypes(
    #     include=["number"]).drop(columns=["Index"],
    #                              errors="ignore")

    # print(f"\nfeatures normalisees =\n{features}")

    # Dico des couleurs pour chaque maison
    house_colors = {
        "Gryffindor": "#e74c3c",
        "Hufflepuff": "#F1C40F",
        "Ravenclaw": "#3498db",
        "Slytherin": "#27ae60",
    }

# ----- MATRICE DE CORRELATION -------------------------

    # Normalisation des notes (ex : toutes entre 0 et 100)
    subjects = [col for col in subjects_norm.columns if col not in [house_col]]
    filtered_data = pd.concat([data[house_col], subjects_norm], axis=1)
    norm_data = filtered_data.copy()
    for subject in subjects:
        norm_data[subject] = round(subjects_norm[subject] * 100, 2)

    high_corr = calculate_correlation(data, 0.8)

    if high_corr.empty:
        print("Aucune paire de variables avec une corrélation élevée trouvée.")
    else:
        # ----- LES PLUS FORTES CORRELATIONS > 0.8 --------------------------
        # Sélection des n premières paires avec une forte corrélation
        n_pairs = len(high_corr)  # On peut mettre un nb en dur
        high_corr_pairs = high_corr.iloc[:n_pairs]  # Sélectionne n 1ere paires

        # Création d'une grille de sous-graphiques
        n_rows = int(np.ceil(n_pairs / 3))  # Nb de lignes (3 col par défaut)
        fig, axes = plt.subplots(
            n_rows, 3, figsize=(15, 10 * n_rows),
            constrained_layout=True,
            )

        # Ajuste `axes` pour qu'il soit tjs une liste, même si 1 seule ligne
        axes = axes.ravel()

        # Parcourir les paires et créer un scatter plot pour chaque
        for idx, (ax, (_, row)) in enumerate(zip(
                                axes, high_corr_pairs.iterrows(),
                                )):
            pc1_col, pc2_col = row["Variable 1"], row["Variable 2"]
            for house, color in house_colors.items():
                subset = norm_data[norm_data[house_col] == house]
                ax.scatter(
                    subset[pc1_col],
                    subset[pc2_col],
                    label=house,
                    color=color,
                    alpha=0.5,
                )
            ax.set_xlabel(pc1_col)
            ax.set_ylabel(pc2_col)
            ax.set_title(f"Corr: {row['Correlation']:.2f}")

        # Supprimer les axes inutilisés si moins de 3 * n_rows paires
        for ax in axes[idx + 1:]:
            fig.delaxes(ax)

        # Sauvegarde et affichage
        plt.suptitle("Scatter plots des paires fortement corrélées")
        plt.legend(title="Maison", loc="upper right", bbox_to_anchor=(1, 0.5))
        fig.canvas.mpl_connect("key_press_event", close_on_enter)
        plt.savefig("./SaveGraph/sp_corr_mosaic")
        plt.show()

        # ----- LA PLUS FORTE CORRELATION ----------------------
        # Sélectionner la première paire pour PC1 et PC2
        pc1_col = high_corr.iloc[0]["Variable 1"]
        pc2_col = high_corr.iloc[0]["Variable 2"]
        # Pour tester différentes colonnes :
        # pc1_col = "Transfiguration"
        # pc2_col = "History of Magic"
        print(f"Axes du graphique : PC1 -> {pc1_col}, PC2 -> {pc2_col}")

        # Visualisation en nuage de points
        plt.figure(figsize=(10, 8))
        for house, color in house_colors.items():
            subset = norm_data[norm_data[house_col] == house]
            plt.scatter(
                subset[pc1_col],
                subset[pc2_col],
                label=house,
                color=color,
                alpha=0.5,
            )

        # Ajouter les axes et le titre
        plt.xlabel(pc1_col)
        plt.ylabel(pc2_col)
        plt.title("Meilleure correlation entre deux matières")
        plt.legend(title="Maison")
        plt.grid(True)

        # Sauvegarde et affichage
        fig = plt.gcf()
        fig.canvas.mpl_connect("key_press_event", close_on_enter)
        plt.savefig("./SaveGraph/sp_corr_unique")
        plt.show()


def main() -> None:
    """Load data and visualize."""
    # On vérifie si l'arg en ligne de commande est fourni
    if len(sys.argv) != 2:
        print("Usage: python scatter_plot.py fichier.csv")
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
        viz_scatterplot(data)
    except KeyboardInterrupt:
        print("\nInterruption du programme par l'utilisateur (Ctrl + C)")
        plt.close("all")  # Ferme tous les graphes ouverts
        sys.exit(0)  # Sort proprement du programme
    except ValueError as e:
        print(e)


if __name__ == "__main__":
    main()
