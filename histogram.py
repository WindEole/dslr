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
import seaborn as sns


def close_on_enter(event: any) -> None:
    """Close the figure when the Enter key is pressed."""
    if event.key == "enter":  # Si la touche 'Enter' est pressée
        plt.close(event.canvas.figure)  # Ferme la figure associée

def viz_histogram(data: pd.DataFrame) -> None:
    """Représente les données sous forme d'histogramme."""
    # Filtrer pour obtenir uniquement les colonnes de notes
    # On cherche une colonne contenant House, avec des variantes
    house_col = None
    for col in data.columns:
        if re.search(r'\b(House|Maison)\b', col, re.IGNORECASE):
            house_col = col
            break

    # Si aucune colonne "House" n'est trouvée, on lève une erreur
    if house_col is None:
        raise ValueError("Aucune colonne 'House' n'a éte trouvée")

    # Supposons que les colonnes de notes soient toutes celles qui restent après avoir exclu les infos personnelles
    subject_columns = [col for col in data.columns if col not in ["Index", house_col, "First Name", "Last Name", "Birthday", "Best Hand", "Arithmancy"]]

    # Créer une nouvelle DataFrame pour les données de chaque maison, en ajoutant une colonne 'Maison' pour différencier
    melted_data = data.melt(id_vars=[house_col], value_vars=subject_columns,
                            var_name="Matière", value_name="Note")

    # Couleurs par maison pour le box plot
    house_colors = {
        "Gryffindor": "red",
        "Slytherin": "green",
        "Hufflepuff": "yellow",
        "Ravenclaw": "blue",
    }

    # Tracer le box plot
    plt.figure(figsize=(15, 8))
    sns.boxplot(x="Matière", y="Note", hue=house_col, data=melted_data, palette=house_colors)

    # Ajuster les axes et la légende
    plt.xlabel("Matières")
    plt.ylabel("Distribution des notes")
    plt.title("Distribution des notes par matière et par maison")
    plt.xticks(rotation=45)  # Rotation des étiquettes pour une meilleure lisibilité
    plt.legend(title="Maisons")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Afficher le graphique
    plt.tight_layout()
    fig = plt.gcf()  # On obtient le graphe en cours
    fig.canvas.mpl_connect("key_press_event", close_on_enter)
    plt.show()



# ----- HISTOGRAMME ----------------------------------
    # # Dictionnaire des maisons et leurs couleurs
    # house_colors = {
    #     "Gryffindor": "#9B2226",  # Rouge
    #     "Hufflepuff": "#FFC300",  # Jaune
    #     "Ravenclaw": "#003B5C",   # Bleu
    #     "Slytherin": "#009900",   # Vert
    # }
    # # On cherche une colonne contenant House, avec des variantes
    # house_col = None
    # for col in data.columns:
    #     if re.search(r'\b(House|Maison)\b', col, re.IGNORECASE):
    #         house_col = col
    #         break

    # # Si aucune colonne "House" n'est trouvée, on lève une erreur
    # if house_col is None:
    #     raise ValueError("Aucune colonne 'House' n'a éte trouvée")

    # # On élague le DataFrame avec seulement les col dont on a besoin
    # # -> les notes
    # note_data = data.select_dtypes(include=["number"])
    # # -> et on ajoute les Maisons
    # filtered_data = pd.concat([data[house_col], note_data], axis=1)

    # # On regroupe par maison et on fait la moyenne pour chaque matière
    # grouped_data = filtered_data.groupby(house_col).mean()

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