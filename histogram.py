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

from describe import ft_max, ft_min, ft_percentile, ft_mean, ft_std, ft_count


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
    print("Effectifs par maison :")
    for maison, count in house_counts.items():
        print(f"{maison}: {count}")

    # Normalisation des notes
    subjects_norm = normalize_data(data)
    print(f"notes normalisées = \n{subjects_norm}")
    # On ajoute les Maisons aux notes
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
    perc_data_min = ft_min(perc_data["Adj_Width"])
    perc_data_max = ft_max(perc_data["Adj_Width"])
    perc_data["Adj_Width"] = (
        min_width + (perc_data["Adj_Width"] - perc_data_min) /
        (perc_data_max - perc_data_min) *
        (max_width - min_width)
    )

# ----- HISTOGRAMME GENERAL ----------------------------------
    # Tracer l'histogramme avec distinction par maison
    plt.figure(figsize=(18, 10))

    subjects_list = perc_data["Subject"].unique()
    houses = perc_data["House"].unique()
    num_houses = len(houses)

    # Calcul de la somme des largeurs de barres pour chaque matière
    widths_sum_per_subject = perc_data.groupby("Subject")["Adj_Width"].sum()

    # Calcul des positions de base cumulatives
    positions_base = np.cumsum([0] + widths_sum_per_subject[:-1].tolist())

    # Couleurs pour chaque plage de percentile pour chaque maison
    # #f5b7b1 #f1948a #ec7063 #e74c3c # rouge clair -> sombre
    # #2e86c1 #3498db #5dade2 #85c1e9 #aed6f1 #bleu sombre -> clair
    # #229954 #27ae60 #52be80 #7dcea0 #a9dfbf # vert sombre -> clair
    percentile_colors = {
        "Gryffindor": ["#e74c3c", "#ec7063", "#f1948a", "#f5b7b1"],  # rouge
        "Hufflepuff": ["#F1C40F", "#F4D03F", "#F7DC6F", "#F9E79F"],  # jaune
        "Ravenclaw": ["#3498db", "#5dade2", "#85c1e9", "#aed6f1"],  # bleu
        "Slytherin": ["#27ae60", "#52be80", "#7dcea0", "#a9dfbf"],  # vert
    }
    edge_colors = {
        "Gryffindor": ["#cb4335"],  # rouge
        "Hufflepuff": ["#d4ac0d"],  # jaune
        "Ravenclaw": ["#2e86c1"],  # bleu
        "Slytherin": ["#229954"],  # vert
    }

    # Dictionnaire pour stocker les maisons déjà tracées dans la légende
    add_legend = {house: False for house in houses}
    add_perc_legend = False

    # Positions de chaque groupe de barres (matières)
    for i, house in enumerate(houses):
        house_data = perc_data[perc_data["House"] == house]
        if house_data.empty:
            print(f"Aucune donnée pour la maison: {house}")
            continue  # Passer à l'itération suivante si aucune donnée

        bar_positions = positions_base + np.arange(len(subjects_list)) + i * (1 / num_houses)

        # Extraction des largeurs de barre pour chq matière dans l'ordre
        house_bar_widths = house_data.set_index("Subject").reindex(subjects_list)["Adj_Width"]

        # Tracer les sections empilées pour chaque plage de percentiles
        for j, subject in enumerate(subjects_list):
            bar_position = bar_positions[j]
            bar_width = house_bar_widths[subject]
            # Extraire les percentiles pour chaque plage (0 calcul sur dict)
            perc_100 = next((item["Percentile"] for item in percentiles if item["House"] == house and item["Subject"] == subject), 0)
            perc_75 = next((item["Percentile"] for item in percentiles_75 if item["House"] == house and item["Subject"] == subject), 0)
            perc_50 = next((item["Percentile"] for item in percentiles_50 if item["House"] == house and item["Subject"] == subject), 0)
            perc_25 = next((item["Percentile"] for item in percentiles_25 if item["House"] == house and item["Subject"] == subject), 0)

            # Vérifier que toutes les valeurs existent avant de tracer
            if None in (perc_100, perc_75, perc_50, perc_25):
                print(f"Données manquantes pour {house} - {subject}")
                continue

            # Déterminer les hauteurs des sections de barre
            heights = [
                perc_25,             # 0 - 25
                perc_50 - perc_25,   # 25 - 50
                perc_75 - perc_50,   # 50 - 75
                perc_100 - perc_75,  # 75 - 100
            ]

            # Tracer chaque section de barre
            bottom = 0
            for k, height in enumerate(heights):
                plt.bar(
                    bar_position,
                    height,
                    width=bar_width,
                    # couleur spécifique au segment :
                    color=percentile_colors[house][k],
                    edgecolor=edge_colors[house],
                    bottom=bottom,  # empiler sur la section précédente
                    # ajouter un label seulement pour la première tranche :
                    label=house if not add_legend[house] and k == 0 else None,
                )
                bottom += height  # Monter le bas pour empiler plage suivante

            # Marquer la maison comme ajoutée à la légende
            add_legend[house] = True

            # Tracer les lignes pour les percentiles 75, 50 et 25
            if perc_75 is not None:
                plt.hlines(
                    y=perc_75,
                    xmin=bar_position - bar_width / 2,
                    xmax=bar_position + bar_width / 2,
                    colors="#273746",
                    label="percentile" if not add_perc_legend else None,
                    linewidth=1.5,
                )
            if perc_50 is not None:
                plt.hlines(
                    y=perc_50,
                    xmin=bar_position - bar_width / 2,
                    xmax=bar_position + bar_width / 2,
                    colors="#273746",
                    linewidth=1.5,
                )
            if perc_25 is not None:
                plt.hlines(
                    y=perc_25,
                    xmin=bar_position - bar_width / 2,
                    xmax=bar_position + bar_width / 2,
                    colors="#273746",
                    linewidth=1.5,
                )
            add_perc_legend = True

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

    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Afficher le graphique
    plt.tight_layout()
    fig = plt.gcf()  # On obtient le graphe en cours
    fig.canvas.mpl_connect("key_press_event", close_on_enter)
    plt.savefig("hist_percentile.png")
    plt.show()
# ----- HISTOGRAMME GENERAL FIN ------------------------------

# ----- HISTOGRAMME MOSAIQUE ---------------------------------
    # Couleurs par maison
    house_colors_new = {
        "Gryffindor": "#e74c3c",
        "Hufflepuff": "#F1C40F",
        "Ravenclaw": "#3498db",
        "Slytherin": "#27ae60",
    }

    # Normalisation des notes (ex : toutes entre 0 et 100)
    subjects = [col for col in subjects_norm.columns if col not in [house_col]]
    norm_data = filtered_data.copy()
    for subject in subjects:
        norm_data[subject] = round(subjects_norm[subject] * 100, 2)

    # Définition de la grille de subplots
    num_subjects = len(subjects)
    cols = 5  # Nombre de colonnes dans la mosaïque
    rows = (num_subjects // cols) + (num_subjects % cols > 0)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10), sharey=True)
    axes = axes.flatten()

    # Créer des histogrammes pour chaque matière
    for idx, subject in enumerate(subjects):
        ax = axes[idx]

        # Histogrammes empilés par maison pour chaque plage de notes
        bins = np.linspace(0, 100, 11)  # Plages de notes de 10 en 10

        # Parcourir chaque maison
        for house, color in house_colors_new.items():
            house_data = norm_data[norm_data[house_col] == house][subject]
            ax.hist(
                house_data, bins=bins, alpha=0.5,
                label=house, color=color, edgecolor="black"
                )

        # Configuration du graphique
        ax.set_title(subject)
        ax.set_xlabel("Plages de notes (%)")
        ax.set_ylabel("Nombre d'élèves")

    # Supprimer les axes vides (si le nombre de matières ne remplit pas la grille)
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    # Légende et affichage
    fig.suptitle("Distribution des notes par matière et par maison")
    plt.legend(title="Maison", loc="upper right", bbox_to_anchor=(1.2, 1))
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Ajuste la disposition
    fig.canvas.mpl_connect("key_press_event", close_on_enter)
    plt.savefig("hist_mosaique.png")
    plt.show()
# ----- HISTOGRAMME MOSAIQUE FIN -----------------------------

    print(f"\nfiltered_data = \n{filtered_data}")

    # COUNT de chaque maison
    # grouped_count_officiel = filtered_data.groupby(house_col).count()
    # print(f"\ncount std =\n{grouped_count_officiel}")
    grouped_count = ft_count(filtered_data.groupby(house_col))
    print(f"\ncount custom=\n{grouped_count}")

    # MEAN de chaque maison
    # grouped_mean_officiel = filtered_data.groupby(house_col).mean()
    # print(f"\nmoyennes std =\n{grouped_mean_officiel}")
    grouped_mean = ft_mean(filtered_data.groupby(house_col))  #, grouped_count)
    print(f"\nmoyennes custom =\n{grouped_mean}")

    # STD de chaque maison
    # grouped_std_officiel = filtered_data.groupby(house_col).std()
    # print(f"\necart-type par Maison =\n{grouped_std_officiel}")
    grouped_std = ft_std(filtered_data.groupby(house_col))  #, grouped_mean, grouped_count)
    print(f"\necart-type par Maison =\n{grouped_std}")

    # COEFF DE VARIATION
    # cv_officiel = grouped_std_officiel / grouped_mean_officiel
    # print(f"\ncoeff de variation =\n{cv}")
    cv = grouped_std / grouped_mean
    print(f"\ncoeff de var custom =\n{cv}")

    # STATS SUR COEFF DE VARIATION
    # cv_count = ft_count(cv)
    # cv_mean = ft_mean(cv)
    cv_std = ft_std(cv)
    cv_max = ft_max(cv)
    cv_min = ft_min(cv)
    print(f"cv max = \n{cv_max}\ncv min = \n{cv_min}")
    range_values = {
        key: (cv_max[key] - cv_min[key])
        for key in cv_max.keys() & cv_min.keys()
        if cv_max[key] is not None and cv_min[key] is not None
    }

    variation_metrics = pd.DataFrame({
        "Std_Dev": cv_std,
        # "Range":cv.max() - cv.min(),
        "Range": range_values,
    })
    # print(f"\nVariation metrics = \n{variation_metrics}")

    sorted_variations = variation_metrics.sort_values(by="Std_Dev")
    # print(f"\nvariations metrics triées croissant :\n{sorted_variations}")

# ---- HISTOGRAMME THREE BEST --------------------------------
    most_homogenous_subjects = sorted_variations.index[:3]
    print("\nmatières les plus homogènes =")
    for i in most_homogenous_subjects:
        print(i)

    # Définition de la grille de subplots
    nb_homogenous_subjects = len(most_homogenous_subjects)
    cols = 5  # Nombre de colonnes dans la mosaïque
    rows = (nb_homogenous_subjects // cols) + (nb_homogenous_subjects % cols > 0)
    fig, axes = plt.subplots(rows, cols, figsize=(10, 5), sharey=True)
    axes = axes.flatten()

    # Créer des histogrammes pour chaque matière
    for idx, subject in enumerate(most_homogenous_subjects):
        ax = axes[idx]

        # Histogrammes empilés par maison pour chaque plage de notes
        bins = np.linspace(0, 100, 11)  # Plages de notes de 10 en 10

        # Parcourir chaque maison
        for house, color in house_colors_new.items():
            house_data = norm_data[norm_data[house_col] == house][subject]
            ax.hist(
                house_data, bins=bins, alpha=0.5,
                label=house, color=color, edgecolor="black",
                )

        # Configuration du graphique
        ax.set_title(subject)
        ax.set_xlabel("Plages de notes (%)")
        ax.set_ylabel("Nombre d'élèves")

    # Supprimer les axes vides (si le nb de matières ne remplit pas la grille)
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    # Légende et affichage
    fig.suptitle("Distribution des notes des 3 matières les plus homogènes")
    plt.legend(title="Maison", loc="upper right", bbox_to_anchor=(1.2, 1))
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Ajuste la disposition
    fig.canvas.mpl_connect("key_press_event", close_on_enter)
    plt.savefig("hist_threebest.png")
    plt.show()
# ---- HISTOGRAMME THREE BEST FIN ----------------------------

# ---- HISTOGRAMME UNIQUE ------------------------------------
    best_subject = sorted_variations.index[0]
    print(f"\nmatière la plus homogène =\n{best_subject}")

    plt.figure(figsize=(10, 6))

    # Histogramme empilé par maison
    bins = np.linspace(0, 100, 11)  # Plages de notes de 10 en 10

    # Parcourir chaque maison
    for house, color in house_colors_new.items():
        house_data = norm_data[norm_data[house_col] == house][best_subject]
        plt.hist(
            house_data, bins=bins, alpha=0.5,
            label=house, color=color, edgecolor="black",
            )

    # Légende
    plt.title(f"Distribution des notes - {best_subject}")
    plt.xlabel("Plages de notes (%)")
    plt.ylabel("Nombre d'élèves")
    plt.legend(title="Maison")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Affichage
    plt.tight_layout()
    fig = plt.gcf()
    fig.canvas.mpl_connect("key_press_event", close_on_enter)
    plt.savefig("hist_unique.png")
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
        print("Usage: python histogram.py fichier.csv")
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
