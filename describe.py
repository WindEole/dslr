"""Describe program.

Ce programme ouvre un fichier de données compressées au format .tgz et
les mets en forme suivant plusieurs types de visualisations graphiques.
"""

import math
import os.path
import shutil
import sys
import tarfile

import pandas as pd


def maximum(val: pd.Series) -> float:
    """Determine the maximum value of a dataset."""
    if not val.empty:  # s'il y a des valeurs non-nulles
        tmp_max = val.iloc[0]  # on initialise avec la première valeur
        for value in val.iloc[1:]:
            if value > tmp_max:
                tmp_max = value
        return tmp_max
    return None  # S'il n'y a pas de valeurs non-nulles


def percentile(rank: float, val: pd.Series, count: float) -> float:
    """Find the percentile of a serie of values.

    param rank - a float value from 0.0 to 100.0
    param val - data (must be sorted)
    return - the percentile of the values.
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
    weight_high = perc_rank - index_low  # Pourcentage pour la valeur haute
    weight_low = 1 - weight_high  # Pourcentage pour la valeur basse

    # Calculer le percentile en pondérant les valeurs inférieure et supérieure
    return round(sort_val.iloc[index_low] * weight_low + sort_val.iloc[index_high] * weight_high, 6)

    # return sort_val.iloc[index]

def minimum(val: pd.Series) -> float:
    """Determine the minimum value of a dataset."""
    if not val.empty:  # s'il y a des valeurs non-nulles
        tmp_min = val.iloc[0]  # on initialise avec la première valeur
        for value in val.iloc[1:]:
            if value < tmp_min:
                tmp_min = value
        return tmp_min
    return None  # S'il n'y a pas de valeurs non-nulles


def standard_deviation(val: pd.Series, mean: float, count: float) -> float:
    """Calculate the standard deviation of a serie of values."""
    if not val.empty:
        diff_mean_val = val - mean  # difference par rapport à la moyenne
        square_diff = diff_mean_val ** 2  # carré des différences
        somme_square_diff = square_diff.sum()  # somme des carrés
        variance = somme_square_diff / (count - 1)  # diviser par count - 1
        std_dev = math.sqrt(variance)
        return std_dev
    return None


def moyenne(val: pd.Series, count: float) -> float:
    """Calculate the mean of all values."""
    if not val.empty:
        return sum(val) / count
        # return round(mean, 6)
    return None


def ft_describe(data: pd.DataFrame) -> None:
    """Reproduit la fonction describe de python.

    Cette fonction prend en paramètre un dataframe pandas, et en extrait :
    :-> count (num + string + time) -> Count number of non-NA/null observations
    :unique(string + time)
    :top (string + time) -> most common value
    :freq (string + time) -> most common value's frequency
    :first (time) -> the first item
    :last (time) -> the last item
    :-> mean (num) -> Mean of the values
    :-> std (num) -> Standard deviation of the observations (= écart-type)
    :-> min (num) -> Minimum of the values in the object
    :-> 25 (num) -> lower percentile
    :-> 50 (num) -> median percentile
    :-> 75 (num) -> upper percentile
    :-> max (num) -> Maximum of the values in the object
    :select_dtypes -> Subset including/excluding columns based on their dtype
    ATTENTION for mixed data types in a dataFrame, the default is to return
    only an analysis of numeric columns !!! (SUJET = NUMERICAL FEATURES ONLY !)
    """
    # On sélectionne uniquement les colonnes numériques
    num_data = data.select_dtypes(include=["number"])

# -----------
    print("\033[91mCi-dessous : la fonction describe officielle :\033[0m")
    print(num_data.describe(include="all"))

    # print("\033[91m\nTest : on imprime les 7 premières colones :\033[0m")
    # test = num_data[~num_data["Charms"].isna()][num_data.columns[1:7]]
    # print(test)

    # print("\033[91m\nTest : puis à partir de la 8ème colonne :\033[0m")
    # test = num_data[~num_data["Charms"].isna()][num_data.columns[8:]]
    # print(test)

    # print("\033[91m\nInfo : impression des dataType des valeurs :\033[0m")
    # print(data.dtypes)
    # for column in num_data.columns:
    #     test = num_data[~num_data[column].isna()][column]
    #     print(test)
    #     print(test.iloc[0])
    #     print("----")
    #     for value in test.iloc[1:4]:
    #         print(value)
# -----------

    # On vérifie s'il y a des colonnes numériques dans le dataFrame
    if num_data.empty:
        print("Aucune colonne numérique dans les données.")
        return

    # 1) COUNT = nb d'observations non nulles
    count_values = {}
    for column in num_data.columns:
        non_null_count = sum(1 for value in num_data[column] if not pd.isnull(value))
        count_values[column] = non_null_count

    # 2) MEAN / STD / MIN / MAX
    mean_values = {}
    std_values = {}
    min_values = {}
    perc_25 = {}
    perc_50 = {}
    perc_75 = {}
    max_values = {}
    for col in num_data.columns:
        # Exclure les valeurs nulles pour le calcul du min
        non_null_values = num_data[~num_data[col].isna()][col]
        # non_null_values = [value for value in num_data[column] if not pd.isnull(value)]
        mean_values[col] = moyenne(non_null_values, count_values[col])
        std_values[col] = standard_deviation(non_null_values, mean_values[col], count_values[col])
        min_values[col] = minimum(non_null_values)
        perc_25[col] = percentile(25, non_null_values, count_values[col])
        perc_50[col] = percentile(50, non_null_values, count_values[col])
        perc_75[col] = percentile(75, non_null_values, count_values[col])
        max_values[col] = maximum(non_null_values)

    # 5) STD -> Standard deviation (= écart-type) EN COURS !!!!
    # std_values = {}
    # for column in num_data.columns:
    #     # Exclure les valeurs nulles pour le calcul du max
    #     non_null_values = [value for value in num_data[column] if not pd.isnull(value)]
    #     if non_null_values:
    #         mean = mean_values[column]
    #     else:
    #         std_values[column] = None

    # IMPRESSION FINALE -------------------------------------------------------
    print("\033[91m\nCi-dessous : ma fonction ft_describe :\033[0m")
    # Ajoutez d'autres statistiques ici
    stats_headers = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']

    # Dictionnaire contenant chaque stat avec les valeurs pour chaque colonne
    stats_data = {
        'count': count_values,
        'mean': mean_values,
        'std': std_values,
        'min': min_values,
        '25%': perc_25,
        '50%': perc_50,
        '75%': perc_75,
        'max': max_values,
    }

    # Récupérer la largeur du terminal
    terminal_width = shutil.get_terminal_size().columns

    # Calculer les largeurs de colonnes dynamiquement
    column_widths = {col: max(len(str(col)), *(len(f"{stats_data[stat].get(col, 'N/A'):.6f}") if isinstance(stats_data[stat].get(col), (int, float)) else len('N/A') for stat in stats_headers)) for col in num_data.columns}

    # Largeur de l'index pour les statistiques
    index_col_width = max(len(stat) for stat in stats_headers) + 2

    # Calculer la largeur totale requise pour toutes les colonnes
    # total_width = index_col_width + sum(column_widths[col] + 2 for col in num_data.columns)

    # Si la largeur dépasse celle du terminal, diviser en blocs
    col_per_block = []
    current_block = []
    current_width = index_col_width

    for col in num_data.columns:
        col_width = column_widths[col] + 2
        if current_width + col_width > terminal_width:
            col_per_block.append(current_block)
            current_block = []
            current_width = index_col_width  # Réinitialiser la largeur du bloc
        current_block.append(col)
        current_width += col_width

    # Ajouter le dernier bloc restant
    if current_block:
        col_per_block.append(current_block)

    # Affichage bloc par bloc
    for block_ind, block_columns in enumerate(col_per_block):
        # En-tête de chaque bloc
        print(f"\n\033[92mBloc {block_ind + 1}/{len(col_per_block)} :\033[0m")

        # Imprimer l'en-tête pour chaque bloc
        header = f"{'':<{index_col_width}}"  # Colonne d'index
        for col in block_columns:
            header += f"{col:>{column_widths[col] + 2}}"
        print(header)

        # Imprimer les valeurs pour chaque statistique dans ce bloc
        for stat in stats_headers:
            row = f"{stat:<{index_col_width}}"  # Nom de la statistique
            for col in block_columns:
                value = stats_data[stat].get(col, 'N/A')
                if isinstance(value, (int, float)):
                    row += f"{value:.6f}".rjust(column_widths[col] + 2)
                else:
                    row += f"{value}".rjust(column_widths[col] + 2)
            print(row)

# SAUVEGARDE AFFICHAGE ---------------------------------------------
    # # Ajoutez d'autres statistiques ici
    # stats_headers = ['count', 'min', 'max', 'mean']

    # # Créez un dictionnaire contenant chaque statistique avec les valeurs pour chaque colonne
    # stats_data = {
    #     'count': count_values,
    #     'mean': mean_values,
    #     'min': min_values,
    #     'max': max_values,
    # }

    # # Calculer les largeurs de colonnes dynamiquement
    # column_widths = {}
    # for col in num_data.columns:
    #     # Trouver la longueur maximale entre le nom de la colonne et les valeurs dans `stats_data`
    #     max_width = max(len(str(col)), *(len(f"{stats_data[stat].get(col, 'N/A'):.6f}") if isinstance(stats_data[stat].get(col), (int, float)) else len('N/A') for stat in stats_headers))
    #     column_widths[col] = max_width

    # # Largeur max de la première colonne (pour les noms des statistiques)
    # index_col_width = max(len(stat) for stat in stats_headers) + 2

    # # Imprimer les en-têtes de colonnes
    # header = f"{'':<{index_col_width}}"  # Colonne d'index vide pour les noms des statistiques
    # for col in num_data.columns:
    #     # header += f"{col:<15}"  # Ajuster la largeur des colonnes de données
    #     header += f"{col:<{column_widths[col] + 2}}"  # Ajustement dynamique + espacement
    # print(header)

    # # Imprimer les valeurs pour chaque statistique
    # for stat in stats_headers:
    #     row = f"{stat:<{index_col_width}}"  # Nom de la statistique (index)
    #     for col in num_data.columns:
    #         value = stats_data[stat].get(col, 'N/A')
    #         # affichage avec 6 décimales et trailing 0 sinon N/A
    #         if isinstance(value, (int, float)):
    #             row += f"{value:.6f}".rjust(column_widths[col] + 2)
    #         else:
    #             row += f"{'N/A':<{column_widths[col] + 2}}"
    #     print(row)


# SAUVEGARDE ANCIEN AFFICHAGE------------------
    # # Déterminer la largeur max de la colonne d'index
    # index_col_width = max(len(col) for col in num_data.columns) + 2

    # # Imprimer les en-tête de colonne (statistiques : count, min, etc...)
    # header = f"{'':<{index_col_width}}"  # Colonne d'index (statistiques)
    # for stat in stats_headers:
    #     header += f"{stat:<15}"
    # print(header)

    # # Imprimer chaque colonne du DataFrame avec ses statistiques
    # for col in num_data.columns:
    #     row = f"{col:<{index_col_width}}"  # Ajustement de la colonne d'index
    #     row += f"{count_values[col]:<15}"
    #     if min_values[col] is not None:
    #         row += f"{min_values[col]:<15}"
    #     else:
    #         row += f"{'N/A':<15}"
    #     if max_values[col] is not None:
    #         row += f"{max_values[col]:<15}"
    #     else:
    #         row += f"{'N/A':<15}"
    #     if mean_values[col] is not None:
    #         row += f"{mean_values[col]:<15}"
    #     else:
    #         row += f"{'N/A':<15}"
    #     print(row)
# SAUVEGARDE ANCIEN AFFICHAGE------------------


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
    for root, dirs, files in os.walk(dir_path):
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
        ft_describe(data)
    else:
        print(f"Erreur : le fichier '{filename}' n'a pas été trouvé.")


if __name__ == "__main__":
    main()