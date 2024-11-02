"""Histogram program.

Ce programme ouvre un fichier de données compressées au format .tgz et
génère une visualisation des données sous forme d'histogramme.
"""

import os.path
import re
import sys
import tarfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def close_on_enter(event: any) -> None:
    """Close the figure when the Enter key is pressed."""
    if event.key == "enter":  # Si la touche 'Enter' est pressée
        plt.close(event.canvas.figure)  # Ferme la figure associée


def ft_percentile(rank: float, val: pd.Series, count: float) -> float:
    """Calculate the desired percentile for a series of values.

    param rank - a float value from 0.0 to 100.0
    param val - data (must be sorted)
    return - the percentile of the values.
    Explication percentile : En-dessous de la valeur retournée se trouvent
    25%, 50% ou 75% des observations.
    """
    if count == 0:  # cas où la série est vide
        return None

    # préparation des paramètre avant calcul
    rank = rank / 100
    sort_val = val.sort_values().reset_index(drop=True)  # Trier et réindexer

    # calcul de l'index percentile (l'index doit rester dans les limites)
    perc_rank = rank * (count - 1)
    index_low = int(perc_rank)  # Partie entière inférieure
    index_high = min(index_low + 1, count - 1)  # Entier sup ou max index

    # Calculer une interpolation si nécessaire
    high = perc_rank - index_low  # Pourcentage pour la valeur haute
    low = 1 - high  # Pourcentage pour la valeur basse

    # Calculer le percentile en pondérant les valeurs inférieure et supérieure
    return sort_val.iloc[index_low] * low + sort_val.iloc[index_high] * high


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


def ft_maximum(val: pd.Series) -> float:
    """Determine the maximum value of a dataset."""
    if not val.empty:  # s'il y a des valeurs non-nulles
        tmp_max = val.iloc[0]  # on initialise avec la première valeur
        for value in val.iloc[1:]:
            if value > tmp_max:
                tmp_max = value
        return tmp_max
    return None  # S'il n'y a pas de valeurs non-nulles


def ft_minimum(val: pd.Series) -> float:
    """Determine the minimum value of a dataset."""
    if not val.empty:  # s'il y a des valeurs non-nulles
        tmp_min = val.iloc[0]  # on initialise avec la première valeur
        for value in val.iloc[1:]:
            if value < tmp_min:
                tmp_min = value
        return tmp_min
    return None  # S'il n'y a pas de valeurs non-nulles


def normalize_data(data: pd.DataFrame, house_col: str) -> pd.Series:
    # On élague le DataFrame avec seulement les col dont on a besoin
    # -> les notes
    subjects = data.select_dtypes(include=["number"]).drop(
        ["Index"],
        axis=1,
        errors="ignore")
    # -> On Normalise les notes (trop disparates !) min-max
    subjects_min = {}
    subjects_max = {}
    for col in subjects.columns:
        non_null_values = subjects[~subjects[col].isna()][col]
        subjects_min[col] = ft_minimum(non_null_values)
        subjects_max[col] = ft_maximum(non_null_values)

    subjects_min_series = pd.Series(subjects_min)
    subjects_max_series = pd.Series(subjects_max)
    subjects_norm = (subjects - subjects_min_series) / (subjects_max_series - subjects_min_series)
    return subjects_norm


def viz_histogram(data: pd.DataFrame) -> None:
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

    house_counts = data[house_col].value_counts()
    print("Effectifs par maison :\n")
    for maison, count in house_counts.items():
        print(f"{maison}: {count}")

    # Dictionnaire des maisons et leurs couleurs
    house_colors = {
        "Gryffindor": "#FF6666",  # Rouge
        "Hufflepuff": "#FFFF66",  # Jaune
        "Ravenclaw": "#3399FF",   # Bleu
        "Slytherin": "#66FF66",   # Vert
    }

    # Normalisation des notes
    subjects_norm = normalize_data(data, house_col)
    # On ajoute les Maisons au notes
    filtered_data = pd.concat([data[house_col], subjects_norm], axis=1)
    # On regroupe par maison
    grouped_data = filtered_data.groupby(house_col)

    rank = 100
    percentiles = calcul_perc(grouped_data, subjects_norm, rank)
    percentiles_75 = calcul_perc(grouped_data, subjects_norm, 75)
    percentiles_50 = calcul_perc(grouped_data, subjects_norm, 50)
    percentiles_25 = calcul_perc(grouped_data, subjects_norm, 25)

    # Créer un DataFrame pour les percentiles
    perc_data = pd.DataFrame(percentiles)

    # Calcul de l'effectif minimum parmi les maisons
    min_effectif = perc_data["Effectif"].min()

    # Création d'une liste des largeurs ajustées (* 0.1 pour ajuster visuelmt)
    perc_data["Adj_Width"] = (perc_data["Effectif"] - min_effectif + 1) * 0.1

    # Effectifs dispersés -> Normalisation des largeurs de barres !
    min_width = 0.05  # largeur min
    max_width = 0.25  # largeur max
    perc_data["Adj_Width"] = (
        min_width + (perc_data["Adj_Width"] - perc_data["Adj_Width"].min()) /
        (perc_data["Adj_Width"].max() - perc_data["Adj_Width"].min()) *
        (max_width - min_width)
    )

# ----- HISTOGRAMME ----------------------------------
    # Tracer l'histogramme avec distinction par maison
    plt.figure(figsize=(18, 10))
    subjects_list = perc_data["Subject"].unique()
    houses = perc_data["House"].unique()

    num_houses = len(houses)

    # Calcul de la somme des largeurs de barres pour chaque matière
    widths_sum_per_subject = perc_data.groupby("Subject")["Adj_Width"].sum()

    # Calcul des positions de base cumulatives
    positions_base = np.cumsum([0] + widths_sum_per_subject[:-1].tolist())

    # Positions de chaque groupe de barres (matières)
    for i, house in enumerate(houses):
        house_data = perc_data[perc_data["House"] == house]
        if house_data.empty:
            print(f"Aucune donnée pour la maison: {house}")
            continue  # Passer à l'itération suivante si aucune donnée

        bar_positions = positions_base + np.arange(len(subjects_list)) + i * (1 / num_houses)

        # Extraction des largeurs de barre pour chq matière dans l'ordre
        house_bar_widths = house_data.set_index("Subject").reindex(subjects_list)["Adj_Width"]

        # Tracer la barre pour chaque maison avec des largeurs ajustées
        plt.bar(
            bar_positions,
            house_data["Percentile"],
            width=house_bar_widths,  # largeur dynamique
            label=house,
            color=house_colors[house],
            # edgecolor="black",
            edgecolor=house_colors[house],
        )
        # Ajouter les lignes pour les percentiles 50 et 25
        for subject in subjects_list:
            # Obtenir la position et la largeur de la barre pour cette matière
            bar_position = bar_positions[subjects_list.tolist().index(subject)]
            bar_width = house_bar_widths[subject]

            # Extraire les percentiles 50 et 25 pour la maison et la matière actuelle
            perc_75_value = next((item["Percentile"] for item in percentiles_75 if item["House"] == house and item["Subject"] == subject), None)
            perc_50_value = next((item["Percentile"] for item in percentiles_50 if item["House"] == house and item["Subject"] == subject), None)
            perc_25_value = next((item["Percentile"] for item in percentiles_25 if item["House"] == house and item["Subject"] == subject), None)

            # Tracer les lignes pour les percentiles 75, 50 et 25
            if perc_75_value is not None:
                plt.hlines(
                    y=perc_75_value,
                    xmin=bar_position - bar_width / 2,
                    xmax=bar_position + bar_width / 2,
                    colors="black",
                    linewidth=1.5,
                )
            if perc_50_value is not None:
                plt.hlines(
                    y=perc_50_value,
                    xmin=bar_position - bar_width / 2,
                    xmax=bar_position + bar_width / 2,
                    colors="black",
                    linewidth=1.5,
                )
            if perc_25_value is not None:
                plt.hlines(
                    y=perc_25_value,
                    xmin=bar_position - bar_width / 2,
                    xmax=bar_position + bar_width / 2,
                    colors="black",
                    linewidth=1.5,
                )

    # Configuration des axes et légende
    plt.xlabel("Subjects")
    plt.ylabel(f"{rank}e Percentile des notes")
    plt.title(f"{rank}e Percentile des notes par matière et par maison")

    scaling = 2.85  # A AJUSTER (position des étiquettes sur abscisse)
    positions_base_scaled = positions_base * scaling
    plt.xticks(
        positions_base_scaled + widths_sum_per_subject.values / 2 * scaling,
        [subject.split()[0] for subject in subjects_list],
        # rotation=45,
        ha="right",
        )

    plt.legend(title="House")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Afficher le graphique
    plt.tight_layout()
    fig = plt.gcf()  # On obtient le graphe en cours
    fig.canvas.mpl_connect("key_press_event", close_on_enter)
    plt.show()
# ----- HISTOGRAMME ----------------------------------


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
        print("Usage: python describe.py fichier.csv")
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
            viz_histogram(data)
        except ValueError as e:
            print(e)
    else:
        print(f"Erreur : le fichier '{filename}' n'a pas été trouvé.")


if __name__ == "__main__":
    main()
