"""Scatter plot program.

Ce programme ouvre un fichier de données compressées au format .tgz et
génère une visualisation des données sous forme de nuage de points.
"""
import os.path
import re
import sys
import tarfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from describe import ft_mean, ft_percentile, ft_std


def close_on_enter(event: any) -> None:
    """Close the figure when the Enter key is pressed."""
    if event.key == "enter":  # Si la touche 'Enter' est pressée
        plt.close(event.canvas.figure)  # Ferme la figure associée


def calcul_perc(
        grouped_data: pd.DataFrame, subjects_norm: list, rank: int) -> list:
    """Calculate the percentiles."""
    percentiles = []
    for house, house_data in grouped_data:
        for subject in subjects_norm:
            non_null_values = house_data[subject].dropna()
            count = len(non_null_values)
            perc_rank = ft_percentile(rank, non_null_values, count)
            percentiles.append({
                "House": house,
                "Subject": subject,
                "Percentile": perc_rank,
                "Effectif": len(non_null_values),
                })
    return percentiles


# def normalize_data(data: pd.DataFrame) -> pd.Series:
#     """Normalize the grades of all subjects between 0.0 and 1.0."""
#     subjects = data.select_dtypes(include=["number"]).drop(
#         ["Index"],
#         axis=1,
#         errors="ignore")
#     # On Normalise les notes (trop disparates !) min-max
#     subjects_min = {}
#     subjects_max = {}
#     for col in subjects.columns:
#         non_null_values = subjects[~subjects[col].isna()][col]
#         subjects_min[col] = ft_minimum(non_null_values)
#         subjects_max[col] = ft_maximum(non_null_values)

#     subjects_min_series = pd.Series(subjects_min)
#     subjects_max_series = pd.Series(subjects_max)
#     subjects_norm = (subjects - subjects_min_series) / (subjects_max_series - subjects_min_series)
#     return subjects_norm


def covariance_matrix(data_centered: pd.DataFrame) -> np.ndarray:
    """Calculate the covariance matrix.

    arg = DataFrame avec des données centrées et réduites
          (moyenne ~ 0 / ecart-type ~ 1)
    """
    # Cov(X,Y) = (1 / (n−1)) * ​n∑i=1 (​(Xi​−Xˉ)(Yi​−Yˉ))
    # Xˉ et Yˉ => moyennes des col X et Y -> proches de 0 car données centrées

    # n = nb d'obvservations (lignes)
    n_samples = len(data_centered)  # nb de lignes

    # on initialise une matrice avec des zero
    covar_matrix = np.zeros((data_centered.shape[1], data_centered.shape[1]))

    # boucle sur chq paire de col pour calculer la covariance entre elles
    for i in range(data_centered.shape[1]):
        for j in range(data_centered.shape[1]):
            # calcul de la covar entre col i et col j
            covar_matrix[i, j] = sum(
                data_centered.iloc[:, i] * data_centered.iloc[:, j]
            ) / (n_samples - 1)

    return covar_matrix


def centering_data(features: pd.DataFrame) -> pd.DataFrame:
    """Center and reduce data.

    centrer les données = soustraire la moyenne de chq col à chq valeur de
        cette col, de sorte que la moyenne de chq col soit nulle.
    réduire les données = diviser chq valeur de la col par l'écart-type de
        cette col, de sorte que chq col ait un écart-type de 1.
    arg = DataFrame only number, cleaned of all NaN
    """
    # comptages des valeurs non-nulles pour chaque colonne
    count_values = {}
    for column in features.columns:
        non_null_count = sum(
            1 for value in features[column] if not pd.isna(value)
        )
        count_values[column] = non_null_count
    # calcul des moyennes et des écarts-types
    mean_values = {}
    std_values = {}
    for col in features.columns:
        non_null_values = features[~features[col].isna()][col]
        mean_values[col] = ft_mean(non_null_values, count_values[col])
        std_values[col] = ft_std(
            non_null_values, mean_values[col], count_values[col],
        )
    data_centered = features.copy()
    for col in features.columns:
        data_centered[col] = (features[col] - mean_values[col]) / std_values[col]
    # Vérification si les données sont bien centrées et réduites :
    # Moyenne : La moyenne de chaque colonne dans data_centered devrait être proche de 0.
    print("Moyennes après centrage et réduction :\n", data_centered.mean())
    # Écart-type : L'écart-type de chaque colonne dans data_centered devrait être proche de 1.
    print("Écarts-types après centrage et réduction :\n", data_centered.std())
    return data_centered


def custom_acp(features: pd.DataFrame) -> pd.DataFrame:
    """Analyse en composants principaux.

    arg = dataFrame numérique, sans NaN
    """
    # 1 / nettoyage, centrage et réduction
    data_centered = centering_data(features)
    # 2 / calcul de la matrice de covariance
    # covar_matrix_np = np.cov(data_centered.T)  # .T de DataFrame = dataFrame transposé
    # print(f"matrice de covariance de np : {covar_matrix_np}")
    covar_matrix = covariance_matrix(data_centered)
    print(f"matrice custom = {covar_matrix}")  # JUSQU'ICI ON EST BON !

    #3 / Calcul des valeurs propres et vecteurs propres
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    return covar_matrix


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

    # # Normalisation des notes
    # subjects_norm = normalize_data(data)
    # # On ajoute les Maisons au notes
    # filtered_data = pd.concat([data[house_col], subjects_norm], axis=1)
    # # On regroupe par maison
    # grouped_data = filtered_data.groupby(house_col)

    # rank = 100
    # percentiles = calcul_perc(grouped_data, subjects_norm, rank)
    # percentiles_75 = calcul_perc(grouped_data, subjects_norm, 75)
    # percentiles_50 = calcul_perc(grouped_data, subjects_norm, 50)
    # percentiles_25 = calcul_perc(grouped_data, subjects_norm, 25)

    # Créer un DataFrame pour les percentiles
    # perc_data = pd.DataFrame(percentiles)

    # # Calcul de l'effectif minimum parmi les maisons
    # min_effectif = perc_data["Effectif"].min()

    # # Création d'une liste des largeurs ajustées (* 0.1 pour ajuster visuelmt)
    # perc_data["Adj_Width"] = (perc_data["Effectif"] - min_effectif + 1) * 0.1

    # # Effectifs dispersés -> Normalisation des largeurs de barres !
    # min_width = 0.05  # largeur min
    # max_width = 0.25  # largeur max
    # perc_data["Adj_Width"] = (
    #     min_width + (perc_data["Adj_Width"] - perc_data["Adj_Width"].min()) /
    #     (perc_data["Adj_Width"].max() - perc_data["Adj_Width"].min()) *
    #     (max_width - min_width)
    # )

# ----- HISTOGRAMME ----------------------------------
#     # Tracer l'histogramme avec distinction par maison
#     plt.figure(figsize=(18, 10))
#     subjects_list = perc_data["Subject"].unique()
    # houses = perc_data["House"].unique()

#     num_houses = len(houses)

#     # Calcul de la somme des largeurs de barres pour chaque matière
#     widths_sum_per_subject = perc_data.groupby("Subject")["Adj_Width"].sum()

#     # Calcul des positions de base cumulatives
#     positions_base = np.cumsum([0] + widths_sum_per_subject[:-1].tolist())


#     # # Couleurs pour chaque plage de percentile pour chaque maison
#     # percentile_colors = {
#     #     "Gryffindor": ["#EF9A9A", "#EF5350", "#E53935", "#C62828"],  # Var rouge
#     #     "Hufflepuff": ["#F9E79F", "#F7DC6F", "#F4D03F", "#F1C40F"],  # Var jaune
#     #     "Ravenclaw": ["#00CCFF", "#0099FF", "#0066FF", "#0033FF"],  # Var bleu
#     #     "Slytherin": ["#C8E6C9", "#81C784", "#4CAF50", "#388E3C"],  # Var vert
#     # }
# #f5b7b1 #f1948a #ec7063 #e74c3c # rouge clair -> sombre
# #2e86c1 #3498db #5dade2 #85c1e9 #aed6f1 #bleu sombre -> clair
# #229954 #27ae60 #52be80 #7dcea0 #a9dfbf # vert sombre -> clair
#     # Couleurs pour chaque plage de percentile pour chaque maison
#     percentile_colors = {
#         "Gryffindor": ["#e74c3c", "#ec7063", "#f1948a", "#f5b7b1"],  # Var rouge
#         # "Gryffindor": ["#B71C1C", "#E53935", "#E57373", "#FFEBEE"],  # Var rouge
#         "Hufflepuff": ["#F1C40F", "#F4D03F", "#F7DC6F", "#F9E79F"],  # Var jaune
#         "Ravenclaw": ["#3498db", "#5dade2", "#85c1e9", "#aed6f1"],  # Var bleu
#         "Slytherin": ["#27ae60", "#52be80", "#7dcea0", "#a9dfbf"],  # Var vert
#     }
#     edge_colors = {
#         "Gryffindor": [ "#cb4335" ],  # Var rouge
#         # "Gryffindor": ["#B71C1C","#B71C1C", "#E53935", "#E57373"],  # Var rouge
#         "Hufflepuff": ["#d4ac0d" ],  # Var jaune
#         # "Hufflepuff": ["#F1C40F", "#F1C40F", "#F4D03F", "#F7DC6F"],  # Var jaune
#         "Ravenclaw": ["#2e86c1"],  # Var bleu
#         # "Ravenclaw": ["#0033FF", "#0033FF", "#0066FF", "#0099FF"],  # Var bleu
#         "Slytherin": ["#229954"],  # Var vert
#         # "Slytherin": ["#388E3C", "#388E3C", "#4CAF50", "#81C784"],  # Var vert
#     }

#     # Dictionnaire pour stocker les maisons déjà tracées dans la légende
#     added_to_legend = {house: False for house in houses}
#     added_perc_legend = False

#     # Positions de chaque groupe de barres (matières)
#     for i, house in enumerate(houses):
#         house_data = perc_data[perc_data["House"] == house]
#         if house_data.empty:
#             print(f"Aucune donnée pour la maison: {house}")
#             continue  # Passer à l'itération suivante si aucune donnée

#         bar_positions = positions_base + np.arange(len(subjects_list)) + i * (1 / num_houses)

#         # Extraction des largeurs de barre pour chq matière dans l'ordre
#         house_bar_widths = house_data.set_index("Subject").reindex(subjects_list)["Adj_Width"]

#     # Tracer les sections empilées pour chaque plage de percentiles
#         for j, subject in enumerate(subjects_list):
#             bar_position = bar_positions[j]
#             bar_width = house_bar_widths[subject]
#             # Extraire les percentiles pour chaque plage (car opération impossible sur dict)
#             perc_100 = next((item["Percentile"] for item in percentiles if item["House"] == house and item["Subject"] == subject), 0)
#             perc_75 = next((item["Percentile"] for item in percentiles_75 if item["House"] == house and item["Subject"] == subject), 0)
#             perc_50 = next((item["Percentile"] for item in percentiles_50 if item["House"] == house and item["Subject"] == subject), 0)
#             perc_25 = next((item["Percentile"] for item in percentiles_25 if item["House"] == house and item["Subject"] == subject), 0)

#             # Vérifier que toutes les valeurs existent avant de tracer
#             if None in (perc_100, perc_75, perc_50, perc_25):
#                 print(f"Données manquantes pour {house} - {subject}")
#                 continue

#             # Déterminer les hauteurs des sections de barre
#             heights = [
#                 perc_25,                  # 0 - 25
#                 perc_50 - perc_25,  # 25 - 50
#                 perc_75 - perc_50,  # 50 - 75
#                 perc_100 - perc_75, # 75 - 100
#             ]

#             # Tracer chaque section de barre
#             bottom = 0
#             for k, height in enumerate(heights):
#                 plt.bar(
#                     bar_position,
#                     height,
#                     width=bar_width,
#                     color=percentile_colors[house][k],  # couleur spécifique au segment
#                     edgecolor=edge_colors[house],
#                     # edgecolor="black",  #if k == 3 else None,  # bordure noire pour la section 75-100 seulement
#                     bottom=bottom,  # empiler sur la section précédente
#                     label=house if not added_to_legend[house] and k == 0 else None,  # ajouter un label seulement pour la première tranche
#                 )
#                 bottom += height  # Monter le bas pour empiler la section suivante

#             # Marquer la maison comme ajoutée à la légende
#             added_to_legend[house] = True

#             # Tracer les lignes pour les percentiles 75, 50 et 25
#             if perc_75 is not None:
#                 plt.hlines(
#                     y=perc_75,
#                     xmin=bar_position - bar_width / 2,
#                     xmax=bar_position + bar_width / 2,
#                     colors="#273746",
#                     # colors="red",
#                     label="percentile" if not added_perc_legend else None,
#                     linewidth=1.5,
#                 )
#             if perc_50 is not None:
#                 plt.hlines(
#                     y=perc_50,
#                     xmin=bar_position - bar_width / 2,
#                     xmax=bar_position + bar_width / 2,
#                     colors="#273746",
#                     # colors="green",
#                     linewidth=1.5,
#                 )
#             if perc_25 is not None:
#                 plt.hlines(
#                     y=perc_25,
#                     xmin=bar_position - bar_width / 2,
#                     xmax=bar_position + bar_width / 2,
#                     colors="#273746",
#                     # colors="blue",
#                     linewidth=1.5,
#                 )
#             added_perc_legend = True

#     # Configuration des axes et légende
#     plt.xlabel("Subjects")
#     plt.ylabel(f"{rank}e Percentile des notes")
#     plt.title(f"{rank}e Percentile des notes par matière et par maison")

#     scaling = 2.85  # A AJUSTER (position des étiquettes sur abscisse)
#     positions_base_scaled = positions_base * scaling
#     plt.xticks(
#         positions_base_scaled + widths_sum_per_subject.values / 2 * scaling,
#         [subject.split()[0] for subject in subjects_list],
#         # rotation=45,
#         ha="right",
#         )

#     # plt.legend(title="House")
#     plt.legend()
#     plt.grid(axis="y", linestyle="--", alpha=0.7)

#     # Afficher le graphique
#     plt.tight_layout()
#     fig = plt.gcf()  # On obtient le graphe en cours
#     fig.canvas.mpl_connect("key_press_event", close_on_enter)
#     plt.savefig("output.png")
#     plt.show()
# ----- HISTOGRAMME ----------------------------------

# ----- SCATTER PLOT (PCA test) ----------------------

    # Supprimer les lignes contenant des NaN
    data_cleaned = data.dropna()

    # Sélectionner les colonnes de données numériques (variables) pour l'ACP
    features = data_cleaned.select_dtypes(include=["number"]).drop(columns=["Index"], errors="ignore")

    # Appliquer l'ACP pour réduire à 2 dimensions
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features)

    # Création d'un DataFrame avec les résultats de l'ACP
    data_pca = pd.DataFrame(data=pca_result, columns=["PC1", "PC2"])
    print(f"Voici le data_pca :\n{data_pca}")

# ----- CUSTOM ACP ------------------------------------



    # pour l'instant on est au centrage :
    data_pca_custom = custom_acp(features)
    print(f"data_custom :\n{data_pca_custom}")



# ----- CUSTOM ACP ------------------------------------

    data_pca[house_col] = data_cleaned[house_col]  # Ajouter les maisons pour les couleurs

    # Dico des couleurs pour chaque maison
    house_colors = {
        "Gryffindor": "#e74c3c",
        "Hufflepuff": "#F1C40F",
        "Ravenclaw": "#3498db",
        "Slytherin": "#27ae60",
    }

    # Visualiser en nuage de points avec une couleur par maison
    plt.figure(figsize=(10, 8))
    for house, color in house_colors.items():
        subset = data_pca[data_pca[house_col] == house]
        plt.scatter(subset["PC1"], subset["PC2"], label=house, color=color, alpha=0.5)

    plt.xlabel("PC1 - Première composante")
    plt.ylabel("PC2 - Deuxième composante")
    plt.title("Nuage de points basé sur l'ACP")
    plt.legend(title="Maison")
    plt.grid(True)
    fig = plt.gcf()  # On obtient le graphe en cours
    fig.canvas.mpl_connect("key_press_event", close_on_enter)
    plt.savefig("acp_direct")
    plt.show()

# ----- BIPLOT : poids des matières dans le nuage de points
    # On applique l'ACP pour biplot
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(features)

    print(f"\033[91mOn explique {round(pca.explained_variance_ratio_.sum() * 100, 2)}"
          " % de la variance totale des données.\033[0m")

    # Coefficients de chargement pour PC1 et PC2
    loadings = pd.DataFrame(
        pca.components_.T,  # Transpose pour avoir les matières en lignes
        columns=["PC1", "PC2"],
        index=features.columns,  # Les colonnes de `data` sont les matières
    )

    # Affichez les matières les plus influentes pour PC1 et PC2
    print("Matières influentes pour PC1 et PC2:")
    print(loadings)

    # Visualisation du biplot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(x_pca[:, 0], x_pca[:, 1], alpha=0.5, c=data_cleaned[house_col].map(house_colors))

    # Ajout des flèches pour chaque matière
    scale_factor = 1000

    # Filtrer les matières avec des charges faibles
    threshold = 0.1  # Seulement garder les matières ayant une influence importante
    significant_loadings = loadings[(np.abs(loadings["PC1"]) > threshold) | (np.abs(loadings["PC2"]) > threshold)]

    for subject in significant_loadings.index:
        ax.arrow(0, 0, significant_loadings.loc[subject, "PC1"] * scale_factor, significant_loadings.loc[subject, "PC2"] * scale_factor,
                 color="r", alpha=0.5, head_width=0.05)
        plt.text(significant_loadings.loc[subject, "PC1"] * scale_factor * 1.15,
                 significant_loadings.loc[subject, "PC2"] * scale_factor * 1.15,
                 subject, color="r", ha="center", va="center")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.title("Biplot des matières dans l'espace des composantes principales")
    plt.grid()
    fig = plt.gcf()  # On obtient le graphe en cours
    fig.canvas.mpl_connect("key_press_event", close_on_enter)
    plt.savefig("acp_biplot")
    plt.show()


def load(path: str) -> pd.DataFrame:
    """Load a file.csv and return a dataset."""
    try:
        data_read = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: The file {path} was not found.")
        return None
    except pd.errors.ParserError:
        print(f"Error: The file {path} is corrupted.")
        return None
    except MemoryError:
        print(f"Error: The file {path} is too large to fit into memory.")
        return None
    except IOError:
        print(f"Error: Unable to open the file at path {path}.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

    data = pd.DataFrame(data_read)

    lines, col = data.shape
    print(f"Loading dataset of dimensions ({lines}, {col})\n")

    return data


def extract_tgz(tgz_file: str, dir_path: str) -> pd.DataFrame:
    """Décompresse un fichier.tgz dans le répertoire.

    :param tgz_file : le chemin du fichier.tgz à décompresser
    :param dir_path : le répertoire où décompresser le fichier.
    """
    try:
        with tarfile.open(tgz_file) as tar:
            tar.extractall(path=dir_path)
            print(f"Fichier {tgz_file} décompressé dans répertoire {dir_path}")
    except Exception as e:
        print(f"Erreur lors de la décompression de {tgz_file}: {e}")
        sys.exit(1)


def find_file(filename: str, dir_path: str):
    """Recherche récursive d'un fichier dans le répertoire donné.

    :param filename: Nom du fichier à rechercher (ex: 'dataset.csv')
    :param dir_path: Répertoire où commencer la recherche
    :return: Chemin complet vers le fichier s'il est trouvé, sinon None
    """
    for root, dir, files in os.walk(dir_path):
        if filename in files:
            return os.path.join(root, filename)

    # si le fichier n'est pas trouvé
    return None


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
