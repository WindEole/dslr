"""Scatter plot program.

Ce programme ouvre un fichier de données compressées au format .tgz et
génère une visualisation des données sous forme de nuage de points.
"""
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from describe import (
    extract_tgz,
    find_file,
    load,
)
from histogram import normalize_data


def close_on_enter(event: any) -> None:
    """Close the figure when the Enter key is pressed."""
    if event.key == "enter":  # Si la touche 'Enter' est pressée
        plt.close(event.canvas.figure)  # Ferme la figure associée


def viz_pairplot(data: pd.DataFrame) -> None:
    """Représente les données sous forme de pair plot."""
    # On cherche une colonne contenant House, avec des variantes
    house_col = None
    for col in data.columns:
        if re.search(r"\b(House|Maison|Houses)\b", col, re.IGNORECASE):
            house_col = col
            break

    # Si aucune colonne "House" n'est trouvée, on lève une erreur
    if house_col is None:
        raise ValueError("Aucune colonne 'House' n'a éte trouvée")

    # Supprimer les lignes contenant des NaN
    data_cleaned = data.dropna()
    # Normalisation des données
    subjects_norm = normalize_data(data_cleaned)
    # On rajoute aux notes normalisées la colonne Maison
    filtered_data = pd.concat([data[house_col], subjects_norm], axis=1)

    # Dico des couleurs pour chaque maison
    house_colors = {
        "Gryffindor": "#e74c3c",
        "Hufflepuff": "#F1C40F",
        "Ravenclaw": "#3498db",
        "Slytherin": "#27ae60",
    }
    pairgrid = sns.pairplot(
        filtered_data,
        hue=house_col,
        palette=house_colors,
        diag_kind="hist",  # Met des histogramme sur la diagonale
        diag_kws={"bins": 10},  # Histogramme avec 10 bins sur la diagonale
        plot_kws={"alpha": 1, "s": 5},  # alpha=Transparence | s=Taille des pts
        )

    # Ajouter des étiquettes personnalisées
    for i, col in enumerate(subjects_norm.columns):
        # Ajustement des étiquettes sur les colonnes
        pairgrid.axes[0, i].set_title(col, fontsize=8, ha="center")
        # Ajustement des étiquettes sur les lignes
        for j in range(len(pairgrid.axes)):
            pairgrid.axes[j, 0].set_ylabel(
                subjects_norm.columns[j].split()[0],
                fontsize=8,
                ha="center",
                )

    sns.move_legend(
        pairgrid,
        "upper center",
        bbox_to_anchor=(0.5, 0.94),
        title="Maison",
        ncol=4,
        markerscale=5,
        )

    # Ajuster les espaces pour éviter le chevauchement des étiquettes
    plt.subplots_adjust(top=0.9, wspace=0.3, hspace=0.3)

    pairgrid.figure.suptitle(
        "Pair plot des matières par Maison",
        fontsize=12,
        y=0.95,
        )
    fig = plt.gcf()
    fig.canvas.mpl_connect("key_press_event", close_on_enter)
    plt.savefig("./SaveGraph/pair_plot")
    plt.show()


def main() -> None:
    """Load data and visualize."""
    # On vérifie si l'arg en ligne de commande est fourni
    if len(sys.argv) != 2:
        print("Usage: python pair_plot.py fichier.csv")
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
            viz_pairplot(data)
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
