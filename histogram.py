"""Histogram program.

Ce programme ouvre un fichier de données compressées au format .tgz et
génère une visualisation des données sous forme d'histogramme.
"""
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from describe import (
    find_file,
    ft_count,
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


def hist_best_subject(
        sorted_variations: pd.DataFrame,
        house_colors_new: dict,
        norm_data: pd.DataFrame,
        house_col: str,
        ) -> None:
    best_subject = sorted_variations.index[0]
    print(f"\nmatière la plus homogène =\n{best_subject}")

    plt.figure(figsize=(10, 6))

    # Histogramme empilé par maison
    bins = np.linspace(0, 100, 11)  # Plages de notes de 10 en 10

    # Parcourir chaque maison
    for house, color in house_colors_new.items():
        house_data = norm_data[norm_data[house_col] == house][best_subject]
        print(f"\nhouse data dans boucle for :\n{house_data}")

        # Calcul de l'histogramme pour la maison
        counts, _ = np.histogram(house_data, bins=bins)

        # Conversion en pourcentage
        total_stud = len(house_data)
        percentages = (counts / total_stud) * 100 if total_stud > 0 else counts

        # Tracer l'histogramme en mode barres empilées
        plt.bar(
            bins[:-1], percentages, width=np.diff(bins), align="edge",
            label=house, color=color, edgecolor="black", alpha=0.5,
        )

    # Légende
    plt.title(f"Distribution des notes - {best_subject} (en %)")
    plt.xlabel("Plages de notes (%)")
    plt.ylabel("Pourcentage d'élèves")
    plt.legend(title="Maison")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Affichage
    plt.tight_layout()
    fig = plt.gcf()
    fig.canvas.mpl_connect("key_press_event", close_on_enter)
    plt.savefig("./SaveGraph/hist_best_perc.png")
    plt.show()


def hist_mosaique(
        subjects: any,
        house_colors_new: dict,
        norm_data: pd.DataFrame,
        house_col: str,
        ) -> None:
    """Visualise les données en histogramme mosaïque."""
    title1 = "Distribution des notes par matière et par maison"
    title2 = "./SaveGraph/hist_mosaique_perc.png"
    if isinstance(subjects, pd.DataFrame):
        subjects = subjects.index[:3]
        print("\nmatières les plus homogènes =")
        for i in subjects:
            print(i)
        title1 = "Distribution des notes des 3 matières les plus homogènes"
        title2 = "./SaveGraph/hist_threebest_perc.png"

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

            # Calcul de l'histogramme
            counts, _ = np.histogram(house_data, bins=bins)
            n_stud = len(house_data)

            # Conversion en pourcentage
            percentages = (counts / n_stud * 100) if n_stud > 0 else counts

            # Tracer les barres empilées
            ax.bar(
                bins[:-1], percentages, width=np.diff(bins), align="edge",
                label=house, color=color, edgecolor="black", alpha=0.5,
            )

        # Configuration du graphique
        ax.set_title(subject)
        ax.set_xlabel("Plages de notes (%)")
        ax.set_ylabel("Pourcentage d'élèves")

    # Supprimer les axes vides (si le nb de matières ne remplit pas la grille)
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    # Légende et affichage
    fig.suptitle(title1)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Ajuste la disposition
    plt.legend(title="Maison", loc="center right", bbox_to_anchor=(2, 0.5))
    fig.canvas.mpl_connect("key_press_event", close_on_enter)
    plt.savefig(title2)
    plt.show()


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
    min_effectif = ft_min(perc_data["Effectif"])

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
                    # color=percentile_colors[house][k],
                    color=percentile_colors[house][k] if k > 0 and k < 3 else "w",
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
    plt.savefig("./SaveGraph/hist_percentile.png")
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

    hist_mosaique(subjects, house_colors_new, norm_data, house_col)

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
    grouped_mean = ft_mean(filtered_data.groupby(house_col))
    print(f"\nmoyennes custom =\n{grouped_mean}")

    # STD de chaque maison
    # grouped_std_officiel = filtered_data.groupby(house_col).std()
    # print(f"\necart-type par Maison =\n{grouped_std_officiel}")
    grouped_std = ft_std(filtered_data.groupby(house_col))
    print(f"\necart-type par Maison =\n{grouped_std}")

    # COEFF DE VARIATION
    # cv_officiel = grouped_std_officiel / grouped_mean_officiel
    # print(f"\ncoeff de variation =\n{cv}")
    cv = grouped_std / grouped_mean
    print(f"\ncoeff de var custom =\n{cv}")

    # STATS SUR COEFF DE VARIATION
    cv_std = ft_std(cv)
    cv_max = ft_max(cv)
    cv_min = ft_min(cv)
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

    sorted_var = variation_metrics.sort_values(by="Std_Dev")
    # print(f"\nvariations metrics triées croissant :\n{sorted_var}")

# ---- HISTOGRAMME THREE BEST --------------------------------
    hist_mosaique(sorted_var, house_colors_new, norm_data, house_col)

# ---- HISTOGRAMME UNIQUE ------------------------------------
    hist_best_subject(sorted_var, house_colors_new, norm_data, house_col)


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
        print(f"Erreur : le fichier '{filename}' n'a pas été trouvé.")
        sys.exit(1)

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


if __name__ == "__main__":
    main()
