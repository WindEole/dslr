"""Scatter plot program.

Ce programme ouvre un fichier de données compressées au format .tgz et
génère une visualisation des données sous forme de nuage de points.
"""
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from describe import (
    extract_tgz,
    find_file,
    ft_max,
    ft_mean,
    ft_min,
    ft_percentile,
    ft_std,
    load,
)


def close_on_enter(event: any) -> None:
    """Close the figure when the Enter key is pressed."""
    if event.key == "enter":  # Si la touche 'Enter' est pressée
        plt.close(event.canvas.figure)  # Ferme la figure associée


# def covariance_matrix(data_centered: pd.DataFrame) -> np.ndarray:
#     """Calculate the covariance matrix.

#     arg = DataFrame avec des données centrées et réduites
#           (moyenne ~ 0 / ecart-type ~ 1)
#     """
#     # Cov(X,Y) = (1 / (n−1)) * ​n∑i=1 (​(Xi​−Xˉ)(Yi​−Yˉ))
#     # Xˉ et Yˉ => moyennes des col X et Y -> proches de 0 car données centrées

#     # n = nb d'obvservations (lignes)
#     n_samples = len(data_centered)  # nb de lignes

#     # on initialise une matrice avec des zero
#     covar_matrix = np.zeros((data_centered.shape[1], data_centered.shape[1]))

#     # boucle sur chq paire de col pour calculer la covariance entre elles
#     for i in range(data_centered.shape[1]):
#         for j in range(data_centered.shape[1]):
#             # calcul de la covar entre col i et col j
#             covar_matrix[i, j] = sum(
#                 data_centered.iloc[:, i] * data_centered.iloc[:, j]
#             ) / (n_samples - 1)

#     return covar_matrix


# def centering_data(features: pd.DataFrame) -> pd.DataFrame:
#     """Center and reduce data.

#     centrer les données = soustraire la moyenne de chq col à chq valeur de
#         cette col, de sorte que la moyenne de chq col soit nulle.
#     réduire les données = diviser chq valeur de la col par l'écart-type de
#         cette col, de sorte que chq col ait un écart-type de 1.
#     arg = DataFrame only number, cleaned of all NaN
#     """
#     mean_values = ft_mean(features)
#     std_values = ft_std(features)

#     data_centered = features.copy()
#     for col in features.columns:
#         data_centered[col] = (features[col] - mean_values[col]) / std_values[col]
#     # Vérification si les données sont bien centrées et réduites :
#     # Moyenne : La moyenne de chq col dans data_centered devrait être ~ 0.
#     print("Moyennes après centrage et réduction :\n", data_centered.mean())
#     # Écart-type : L'écart-type de chq col dans data_centered devrait être ~ 1.
#     print("Écarts-types après centrage et réduction :\n", data_centered.std())
#     return data_centered


# def custom_acp(features: pd.DataFrame) -> pd.DataFrame:
#     """Analyse en composants principaux.

#     arg = dataFrame numérique, sans NaN
#     """
#     # 1 / nettoyage, centrage et réduction
#     data_centered = centering_data(features)
#     # 2 / calcul de la matrice de covariance
#     # covar_matrix_np = np.cov(data_centered.T)  # .T de DataFrame = dataFrame transposé
#     # print(f"matrice de covariance de np : {covar_matrix_np}")
#     covar_matrix = covariance_matrix(data_centered)
#     print(f"matrice custom = {covar_matrix}")  # JUSQU'ICI ON EST BON !

#     #3 / Calcul des valeurs propres et vecteurs propres
#     eigenvalues, eigenvectors = np.linalg.eigh(covar_matrix)

#     return covar_matrix


def extract_high_corr(corr_matrix: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """Extract pairs of columns whose correlation is near 1 or -1.

    Argument:
        corr_matrix(pd.DataFrame): matrice de correlation.
        threshold (float): seuil de correlation absolue (0.9 par défaut).

    Retourne pd.DataFrame listant les paires de colonnes et leur corrélation.
    """
    # Convertir la matrice en format long (simplifie filtrage, manip et présentation)
    corr_long = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
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
    corr_matrix = pd.DataFrame(corr_matrix, index=data_cr.columns, columns=data_cr.columns)
    return corr_matrix


def normalize_data(data: pd.DataFrame) -> pd.Series:
    """Normalize the grades of all subjects between 0.0 and 1.0."""
    subjects = data.select_dtypes(include=["number"]).drop(
        ["Index"],
        axis=1,
        errors="ignore")
    # On Normalise les notes (trop disparates !) min-max
    subjects_max = ft_max(subjects)
    subjects_min = ft_min(subjects)
    s_min_series = pd.Series(subjects_min)
    s_max_series = pd.Series(subjects_max)
    subjects_norm = (subjects - s_min_series) / (s_max_series - s_min_series)
    return subjects_norm


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
    print("Effectifs par maison :\n")
    for maison, count in house_counts.items():
        print(f"{maison}: {count}")

# ----- SCATTER PLOT (PCA test) ----------------------

    # Supprimer les lignes contenant des NaN
    data_cleaned = data.dropna()

    subjects_norm = normalize_data(data_cleaned)

    # Sélectionner les colonnes de données numériques (variables) pour l'ACP
    features = subjects_norm.select_dtypes(
        include=["number"]).drop(columns=["Index"], errors="ignore",
        )

    print(f"\nfeatures normalisees =\n{features}")

# # ----- ACP PYTHON SKLEARN -----------------------------

#     # Appliquer l'ACP pour réduire à 2 dimensions
#     pca = PCA(n_components=2)
#     pca_result = pca.fit_transform(features)

#     # Création d'un DataFrame avec les résultats de l'ACP
#     data_pca = pd.DataFrame(data=pca_result, columns=["PC1", "PC2"])
#     print(f"Voici le data_pca :\n{data_pca}")


# # ----- CUSTOM ACP ------------------------------------

# # L'objectif est d'obtenir un data_pca équivalent au PCA de sklearn
#     # data_pca_custom = custom_acp(features)
#     # print(f"data_custom :\n{data_pca_custom}")

# # ----- CUSTOM ACP FIN ---------------------------------

#     data_pca[house_col] = data_cleaned[house_col]  # Ajouter les maisons pour les couleurs

    # Dico des couleurs pour chaque maison
    house_colors = {
        "Gryffindor": "#e74c3c",
        "Hufflepuff": "#F1C40F",
        "Ravenclaw": "#3498db",
        "Slytherin": "#27ae60",
    }

#     # Visualiser en nuage de points avec une couleur par maison
#     plt.figure(figsize=(10, 8))
#     for house, color in house_colors.items():
#         subset = data_pca[data_pca[house_col] == house]
#         plt.scatter(subset["PC1"], subset["PC2"], label=house, color=color, alpha=0.5)

#     plt.xlabel("PC1 - Première composante")
#     plt.ylabel("PC2 - Deuxième composante")
#     plt.title("Nuage de points basé sur l'ACP")
#     plt.legend(title="Maison")
#     plt.grid(True)
#     fig = plt.gcf()  # On obtient le graphe en cours
#     fig.canvas.mpl_connect("key_press_event", close_on_enter)
#     plt.savefig("acp_direct")
#     # plt.show()

# # ----- BIPLOT : poids des matières dans le nuage de points
#     print(f"\033[91mOn explique {round(pca.explained_variance_ratio_.sum() * 100, 2)}"
#           " % de la variance totale des données.\033[0m")

#     # Coefficients de chargement pour PC1 et PC2
#     loadings = pd.DataFrame(
#         pca.components_.T,  # Transpose pour avoir les matières en lignes
#         columns=["PC1", "PC2"],
#         index=features.columns,  # Les colonnes de `data` sont les matières
#     )

#     # Affichez les matières les plus influentes pour PC1 et PC2
#     print("Matières influentes pour PC1 et PC2:")
#     print(loadings)

#     # Visualisation du biplot
#     fig, ax = plt.subplots(figsize=(10, 8))
#     ax.scatter(pca_result[:, 0], pca_result[:, 1],
#         alpha=0.5, c=data_cleaned[house_col].map(house_colors))

#     # Ajout des flèches pour chaque matière
#     scale_factor = 1

#     # Filtrer les matières avec des charges faibles
#     threshold = 0.1  # Seulement garder les matières ayant une influence importante
#     significant_load = loadings[(np.abs(loadings["PC1"]) > threshold) | (np.abs(loadings["PC2"]) > threshold)]

#     print(f"vecteurs ? =\n{significant_load}")

#     for subject in significant_load.index:
#         ax.arrow(0, 0,
#                  significant_load.loc[subject, "PC1"] * scale_factor,
#                  significant_load.loc[subject, "PC2"] * scale_factor,
#                  color="r", alpha=0.5, head_width=0.05)
#         plt.text(significant_load.loc[subject, "PC1"] * scale_factor * 1.15,
#                  significant_load.loc[subject, "PC2"] * scale_factor * 1.15,
#                  subject, color="r", ha="center", va="center")

#     ax.set_xlabel("PC1")
#     ax.set_ylabel("PC2")
#     plt.title("Biplot des matières dans l'espace des composantes principales")

#     # Ajout manuel de la légende pour les maisons
#     legend_handles = [
#         plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=10, label=house)
#         for house, color in house_colors.items()
#     ]
#     ax.legend(handles=legend_handles, title="Maison", loc="best")

#     plt.grid()
#     fig = plt.gcf()  # On obtient le graphe en cours
#     fig.canvas.mpl_connect("key_press_event", close_on_enter)
#     plt.savefig("acp_biplot")
#     plt.show()

# ----- MATRICE DE CORRELATION -------------------------

    # Normalisation des notes (ex : toutes entre 0 et 100)
    subjects = [col for col in subjects_norm.columns if col not in [house_col]]
    filtered_data = pd.concat([data[house_col], subjects_norm], axis=1)
    norm_data = filtered_data.copy()
    for subject in subjects:
        norm_data[subject] = round(subjects_norm[subject] * 100, 2)

    # Test avec la méthode corr
    corr_matrix = ft_corr(features)
    print(f"\nmethode corr: \n{features.corr()}")
    print(f"\nmethode corr custom: \n{corr_matrix}")
    high_corr = extract_high_corr(corr_matrix)
    print(f"\nHigh correlation pair :\n{high_corr}")

    if high_corr.empty:
        print("Aucune paire de variables avec une corrélation élevée trouvée.")
    else:
# ----- LES PLUS FORTES CORRELATIONS > 0.8 --------------------------
        # Sélection des n premières paires avec une forte corrélation
        n_pairs = len(high_corr)  # On peut mettre un nb en dur
        high_corr_pairs = high_corr.iloc[:n_pairs]  # Sélectionne n 1ere paires

        # Création d'une grille de sous-graphiques
        n_rows = int(np.ceil(n_pairs / 3))  # Nombre de lignes (3 colonnes par défaut)
        fig, axes = plt.subplots(
            n_rows, 3, figsize=(15, 10 * n_rows),
            constrained_layout=True,
            )

        # Ajuste `axes` pour qu'il soit tjs une liste, même si 1 seule ligne
        axes = axes.ravel()

        # Parcourir les paires et créer un scatter plot pour chaque
        for idx, (ax, (_, row)) in enumerate(zip(axes, high_corr_pairs.iterrows())):
            pc1_col, pc2_col = row["Variable 1"], row["Variable 2"]  # Paires de colonnes
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

        # Ajout d'une légende globale
        # handles, labels = ax.get_legend_handles_labels()
        # fig.legend(handles, labels, title="Maison", loc="upper center", ncol=4)

        # Sauvegarde et affichage
        plt.suptitle("Scatter plots des paires fortement corrélées")
        plt.legend(title="Maison", loc="upper right", bbox_to_anchor=(1, 0.5))
        fig.canvas.mpl_connect("key_press_event", close_on_enter)
        plt.savefig("correlation_mosaic")
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
            subset = norm_data[norm_data[house_col] == house]  # Filtrer par catégorie
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
        plt.savefig("correlation")
        plt.show()


def main() -> None:
    """Load data and visualize."""
    # On vérifie si l'arg en ligne de commande est fourni
    if len(sys.argv) != 2:
        print("Usage: python scatterplot.py fichier.csv")
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
            viz_scatterplot(data)
        except KeyboardInterrupt:
            print("\nInterruption du programme par l'utilisateur (Ctrl + C)")
            plt.close("all")  # Ferme tous les graphes ouverts
            sys.exit(0)  # Sort proprement du programme
        except ValueError as e:
            print(e)
    else:
        print(f"Erreur : le fichier '{filename}' n'a pas été trouvé.")


if __name__ == "__main__":
    main()
