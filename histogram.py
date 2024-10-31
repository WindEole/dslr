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


# def ft_percentile(rank: float, val: pd.Series, count: float) -> float:
#     """Find the percentile of a serie of values.

#     param rank - a float value from 0.0 to 100.0
#     param val - data (must be sorted)
#     return - the percentile of the values.
#     Explication percentile : En-dessous de la valeur retournée se trouvent
#     25%, 50% ou 75% des observations.
#     """
#     if count == 0:  # cas où la série est vide
#         return None

#     # préparation des paramètre avant calcul
#     rank = rank / 100
#     sort_val = val.sort_values().reset_index(drop=True)  # Trier et réindexer

#     # calcul de l'index percentile (l'index doit rester dans les limites)
#     perc_rank = rank * (count - 1)
#     index_low = int(perc_rank)  # Partie entière inférieure
#     index_high = min(index_low + 1, count - 1)  # Entier sup ou max index

#     # Calculer une interpolation si nécessaire
#     high = perc_rank - index_low  # Pourcentage pour la valeur haute
#     low = 1 - high  # Pourcentage pour la valeur basse

#     # Calculer le percentile en pondérant les valeurs inférieure et supérieure
#     return sort_val.iloc[index_low] * low + sort_val.iloc[index_high] * high


def ft_percentile(rank, values):
    """Calculate the desired percentile for a series of values."""
    sorted_values = values.sort_values()
    index = int(rank / 100 * len(values) - 0.5)
    return sorted_values.iloc[index] if len(values) > 0 else None


def viz_histogram(data: pd.DataFrame) -> None:
    """Représente les données sous forme d'histogramme."""
    # On cherche une colonne contenant House, avec des variantes
    house_col = None
    for col in data.columns:
        if re.search(r'\b(House|Maison)\b', col, re.IGNORECASE):
            house_col = col
            break

    # Si aucune colonne "House" n'est trouvée, on lève une erreur
    if house_col is None:
        raise ValueError("Aucune colonne 'House' n'a éte trouvée")

    house_counts = data[house_col].value_counts()
    print("Effectifs par maison :")
    print(house_counts)

# ----- HISTOGRAMME ----------------------------------
    # Dictionnaire des maisons et leurs couleurs
    house_colors = {
        "Gryffindor": "#CC0000",  # Rouge
        "Hufflepuff": "#FFCC00",  # Jaune
        "Ravenclaw": "#000099",   # Bleu
        "Slytherin": "#009900",   # Vert
    }

    # On élague le DataFrame avec seulement les col dont on a besoin
    # -> les notes
    subjects = data.select_dtypes(include=["number"]).drop(["Index"], axis=1, errors='ignore')
    # -> On Normalise les notes (trop disparates !) min-max
    subjects_norm = (subjects - subjects.min()) / (subjects.max() - subjects.min())
    # -> et on ajoute les Maisons
    filtered_data = pd.concat([data[house_col], subjects_norm], axis=1)
    print("\nDonnées filtrées et normalisées : \n")
    print(filtered_data)

    # On regroupe par maison
    grouped_data = filtered_data.groupby(house_col)

    # Percentile 75 par maison
    percentiles = []
    rank = 75
    for house, house_data in grouped_data:
        for subject in subjects_norm:
            non_null_values = house_data[subject].dropna()
            perc_rank = ft_percentile(rank, non_null_values)
            percentiles.append({
                "House": house,
                "Subject": subject,
                "Percentile": perc_rank,
                "Effectif": len(non_null_values),
                })

    # Créer un DataFrame pour les percentiles
    perc_data = pd.DataFrame(percentiles)
    print("\nPercentile data = \n")
    print(perc_data)

    # Calcul de l'effectif minimum parmi les maisons
    min_effectif = perc_data["Effectif"].min()
    print(min_effectif)

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

    # Tracer l'histogramme avec distinction par maison
    plt.figure(figsize=(18, 10))
    # bar_width = 0.1  # largeur de base des barres
    subjects_list = perc_data["Subject"].unique()
    houses = perc_data["House"].unique()

    # Calcul des positions des barres
    num_houses = len(houses)
    bar_widths = perc_data.groupby("House")["Adj_Width"].apply(list).to_dict()
    positions_base = np.arange(len(subjects_list)) * (1 + max_width)

    # calcul des largeurs max par matière
    # max_widths_per_subject = [
    #     perc_data[perc_data["Subject"] == subject]["Adj_Width"].max()
    #     for subject in subjects_list
    # ]
    # calcul des positions de base ajustées avec les largeurs max
    # positions_base = np.arange(len(subjects_list)) * 1.5  # espacement plus gd
    # positions_base = np.cumsum([0] + max_widths_per_subject[:-1]) + np.arange(len(subjects_list))

    # Positions de chaque groupe de barres (matières)
    for i, house in enumerate(houses):
        house_data = perc_data[perc_data["House"] == house]
        if house_data.empty:
            print(f"Aucune donnée pour la maison: {house}")
            continue  # Passer à l'itération suivante si aucune donnée

        # bar_positions = [pos + i * 0.25 for pos in positions_base]

        # Affichage des informations pour déboguer
        print(f"\nMaison: {house}")
        print(f"Nombre de matières: {len(subjects_list)}, Nombre de valeurs: {len(house_data)}")

        # bar_positions = positions_base + (i * (1 + max_width))
        bar_positions = np.arange(len(subjects_list)) + i * (1 / num_houses)

        # # Initialisation des bar_positions pour chaque maison
        # bar_positions = positions_base.copy().astype(float)
        # # Calculer l'offset basé sur les largeurs des barres précédentes
        # if i > 0:
        #     # Ajouter les largeurs des barres de la maison précédente
        #     # prev_house = houses[i - 1]
        #     prev_house_data = perc_data[perc_data["House"] == houses[i - 1]]
        #     if prev_house_data.empty:
        #         print(f"Aucune donnée pour la maison précédente : {house[i - 1]}")
        #         continue
        #     # Vérifiez la forme des données avant l'addition
        #     print(f"previous_house_data['Adj_Width'].values shape: {prev_house_data['Adj_Width'].values.shape}")
        #     bar_positions += prev_house_data["Adj_Width"].values

        # Extraction des largeurs de barre pour chq matière dans l'ordre
        house_bar_widths = house_data.set_index("Subject").reindex(subjects_list)["Adj_Width"]
        print(f"\nMaison = {house}")
        print(f"largeurs de barre = \n{house_bar_widths}")

        # Tracer la barre pour chaque maison avec des largeurs ajustées
        plt.bar(
            bar_positions,
            house_data["Percentile"],
            width=house_bar_widths,  # largeur dynamique
            label=house,
            color=house_colors[house],
            edgecolor="black",
        )

    # Configuration des axes et légende
    plt.xlabel("Subjects")
    plt.ylabel(f"{rank}e Percentile des notes")
    plt.title(f"{rank}e Percentile des notes par matière et par maison")
    plt.xticks(positions_base + 0.25, [subject.split()[0] for subject in subjects_list], rotation=45)
    plt.legend(title="House")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Afficher le graphique
    plt.tight_layout()
    fig = plt.gcf()  # On obtient le graphe en cours
    fig.canvas.mpl_connect("key_press_event", close_on_enter)
    plt.show()



    # # Normalisation des données
    # normalized_data = (grouped_data - grouped_data.min()) / (grouped_data.max() - grouped_data.min())

    # subjects = normalized_data.columns  # liste des matières
    # n_subjects = len(subjects)  # nb de matières
    # bar_width = 0.2  # largeur des barres
    # x = np.arange(n_subjects)  # position des barres sur l'axe x

    # plt.figure(figsize=(12, 6))  # taille du graphique

    # # Ajout des barres pour chq maison
    # for i, house in enumerate(house_colors.keys()):
    #     plt.bar(x + (i * bar_width),
    #             normalized_data.loc[house],
    #             bar_width,
    #             label=house,
    #             color=house_colors[house]
    #             )

    # # On recupere juste le 1er mot de chaque matiere pour les etiquettes
    # short_subjects = [subjects.split()[0] for subject in subjects]

    # # Ajout des étiquettes et du titre
    # plt.xlabel('Subjects')
    # plt.ylabel("Moyenne des notes")
    # plt.title("Moyennes des notes par matière et par maison")
    # # Centrer les étiquettes des matières
    # plt.xticks(x + bar_width * (len(house_colors) - 1) / 2, short_subjects)
    # plt.legend(title='Houses')
    # # plt.ylim(0, 100)  # A AJUSTER en fct des notes max
    # # Grille pour la visibilité
    # plt.grid(axis='y', linestyle='--', alpha=0.7)
    # # Ajuster la mise en page
    # plt.tight_layout()
    # fig = plt.gcf()  # On obtient le graphe en cours
    # fig.canvas.mpl_connect("key_press_event", close_on_enter)
    # plt.show()
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
        viz_histogram(data)
    else:
        print(f"Erreur : le fichier '{filename}' n'a pas été trouvé.")


if __name__ == "__main__":
    main()