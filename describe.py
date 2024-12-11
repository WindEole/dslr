"""Describe program.

Ce programme ouvre un fichier de données compressées au format .tgz et
génère des statistiques descriptives à partir des données.
"""
import math
import os.path
import shutil
import sys
import tarfile

import numpy as np
import pandas as pd


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


def ft_std(val: any) -> float:
    """Calculate the standard deviation of a serie of values.

    Parameter:
        val: a pd.Series or ad DataFrameGroupBy object
        mean: a float or a DataFrame
        count: a int/float or a DataFrame
    Returns:
        if val is pd.Series, mean & count = float -> float
        if val is DataFrameGroupBy -> dict to DataFrame with group names as
        keys and a sub-dict of columns standard deviation (float) as values
    """
    if isinstance(val, pd.DataFrame):
        std_values = {}
        mean = ft_mean(val)
        count = ft_count(val)
        for col in val.columns:
            non_null_values = val[col].dropna()
            if not non_null_values.empty:
                diff_mean_val = non_null_values - mean[col]  # difference par rapport à la moyenne
                square_diff = diff_mean_val ** 2  # carré des différences
                somme_square_diff = square_diff.sum()  # somme des carrés
                variance = somme_square_diff / (count[col] - 1) if count[col] > 1 else 0  # diviser par count - 1
                std_values[col] = math.sqrt(variance)
            else:
                std_values[col] = None
        return std_values
    elif isinstance(val, pd.core.groupby.DataFrameGroupBy):
        std_values = {}
        mean_values = ft_mean(val)
        count_values = ft_count(val)
        for group_name, group_df in val:
            num_columns = group_df.select_dtypes(include=[np.number])
            mean = mean_values.loc[group_name]
            count = count_values.loc[group_name]
            diff_mean_val = num_columns - mean
            square_diff = diff_mean_val ** 2
            somme_square_diff = square_diff.sum()
            variance = somme_square_diff / (count - 1)
            std_values[group_name] = variance.apply(np.sqrt)
        return pd.DataFrame(std_values).T
    else:
        raise TypeError("Input must be a DataFrame or DataFrameGroupBy")


def ft_max(val: any) -> float:
    """Determine the maximum value in a dataframe, series or dictionary."""
    if isinstance(val, pd.DataFrame):  #------------------- DATAFRAME
        max_values = {}
        for col in val.columns:
            non_null_values = val[col].dropna()
            if not non_null_values.empty:
                tmp_max = non_null_values.iloc[0]  # init avec la 1ère valeur
                for value in non_null_values.iloc[1:]:
                    if value > tmp_max:
                        tmp_max = value
                max_values[col] = tmp_max
            else:
                max_values[col] = None
        return max_values

    elif isinstance(val, pd.Series):  #----------------------- SERIES
        if not val.empty:  # s'il y a des valeurs non-nulles
            tmp_max = val.iloc[0]  # initialise avec la 1ière valeur
            for value in val.iloc[1:]:
                if value > tmp_max:
                    tmp_max = value
            return tmp_max
        return None  # S'il n'y a pas de valeurs non-nulles

    elif isinstance(val, dict):  #------------------------ DICTIONARY
        # Si un dictionnaire est fourni
        if not val:  # Vérifie si le dictionnaire est vide
            return None
        # Initialisation avec la première paire clé-valeur non None
        iterator = ((k, v) for k, v in val.items() if v is not None)
        try:
            tmp_max_key, tmp_max_value = next(iterator)  # Première clé-valeur
        except StopIteration:
            return None  # Tous les éléments sont None
        # Parcours des autres éléments
        for k, v in iterator:
            if v is None:  # Ignore les valeurs None
                continue
            compare_value = v
            if compare_value > tmp_max_value:
                tmp_max_value = compare_value
                tmp_max_key = k
        return tmp_max_key
    else:
        raise TypeError("Input must be a Series, DataFrame, or dictionary.")


def ft_min(val: any) -> float:
    """Determine the minimum value of a dataset."""
    if isinstance(val, pd.DataFrame):
        min_values = {}
        for col in val.columns:
            non_null_values = val[col].dropna()
            if not non_null_values.empty:
                tmp_min = non_null_values.iloc[0]  # init avec la 1ere valeur
                for value in non_null_values.iloc[1:]:
                    if value < tmp_min:
                        tmp_min = value
                min_values[col] = tmp_min
            else:
                min_values[col] = None
        return min_values
    elif isinstance(val, pd.Series):
        if not val.empty:  # s'il y a des valeurs non-nulles
            tmp_min = val.iloc[0]  # on initialise avec la première valeur
            for value in val.iloc[1:]:
                if value < tmp_min:
                    tmp_min = value
            return tmp_min
        return None  # S'il n'y a pas de valeurs non-nulles
    else:
        raise TypeError("Input must be a Series or DataFrame")


def ft_mean(val: any) -> any:
    """Calculate the mean of all values.

    Parameter:
        val: a DataFrame or a DataFrameGroupBy object
    Returns:
        if val is a DataFrame -> dict
        if val is a DataFrameGroupBy -> dict to dataFrame with group names as
        keys and a sub-dict of columns means (float) as values
    """
    if isinstance(val, pd.DataFrame):
        mean_values = {}
        for col in val.columns:
            non_null_values = val[col].dropna()
            if not non_null_values.empty:
                mean_values[col] = non_null_values.sum() / len(non_null_values)
            else:
                mean_values[col] = None  # gestion des colonnes vides
        return mean_values
    elif isinstance(val, pd.core.groupby.DataFrameGroupBy):
        mean_values = {}
        for group_name, group_df in val:
            count = ft_count(val)
            num_columns = group_df.select_dtypes(include=[np.number])
            mean_values[group_name] = num_columns.sum() / count.loc[group_name]
        return pd.DataFrame(mean_values).T
    else:
        raise TypeError("Invalid input types for val or count.")


def ft_count(val: any) -> pd.DataFrame:
    """Count the number of non null values.

    Parameter:
        val: a DataFrame or a DataFrameGroupBy object
    Returns:
        if val is a DataFrame -> dictionary of columns counts
        if val is a DataFrameGroupBy -> dict (to dataFrame) with group names
        as keys and a sub-dictionnary of column counts as values
    """
    if isinstance(val, pd.DataFrame):
        count_values = {}
        for column in val.columns:
            non_null_count = sum(
                1 for value in val[column] if not pd.isna(value)
            )
            count_values[column] = non_null_count
        return count_values
    elif isinstance(val, pd.core.groupby.DataFrameGroupBy):
        count_values = {
            group_name: group_df.notna().sum()
            for group_name, group_df in val
        }
        return pd.DataFrame(count_values).T
    else:
        raise TypeError("Input must be a DataFrame or a DataFrameGroupBy object")


def ft_describe(data: pd.DataFrame) -> pd.DataFrame:
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
    count_values = ft_count(num_data)

    # 2) MEAN / STD / MIN / MAX
    mean_values = ft_mean(num_data)
    std_values = ft_std(num_data)
    min_values = ft_min(num_data)
    max_values = ft_max(num_data)

    # 3) PERCENTILES
    perc_25 = {}
    perc_50 = {}
    perc_75 = {}
    for col in num_data.columns:
        # Exclure les valeurs nulles pour le calcul du min
        non_null_values = num_data[~num_data[col].isna()][col]
        perc_25[col] = ft_percentile(25, non_null_values, count_values[col])
        perc_50[col] = ft_percentile(50, non_null_values, count_values[col])
        perc_75[col] = ft_percentile(75, non_null_values, count_values[col])

    # IMPRESSION FINALE -------------------------------------------------------
    print("\033[91m\nCi-dessous : ma fonction ft_describe :\033[0m")
    # Ajoutez d'autres statistiques ici
    stats_headers = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]

    # Dictionnaire contenant chaque stat avec les valeurs pour chaque colonne
    stats_data = {
        "count": count_values,
        "mean": mean_values,
        "std": std_values,
        "min": min_values,
        "25%": perc_25,
        "50%": perc_50,
        "75%": perc_75,
        "max": max_values,
    }

    # ----- AFFICHAGE SUR LE TERMINAL -----------------------------------------

    # Récupérer la largeur du terminal
    terminal_width = shutil.get_terminal_size().columns

    # Calculer les largeurs de colonnes dynamiquement
    column_widths = {}
    for col in num_data.columns:
        # Largeur de base : le nom de la colonne
        max_width = len(str(col))

        # Calcul de la largeur pour chq stat
        for stat in stats_headers:
            value = stats_data[stat].get(col, "N/A")
            if isinstance(value, (int, float)):
                width = len(f"{value:.6f}")  # Format 6 décimales
            else:
                width = len(str(value))

        # Màj max_width si la largeur de la stat est plus grande
        max_width = max(max_width, width)

        # Stocker la largeur max trouvée pour cette colonne
        column_widths[col] = max_width + 2

    # Largeur de l'index pour les statistiques
    index_col_width = max(len(stat) for stat in stats_headers) + 2

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
                value = stats_data[stat].get(col, "N/A")
                if isinstance(value, (int, float)):
                    row += f"{value:.6f}".rjust(column_widths[col] + 2)
                else:
                    row += f"{value}".rjust(column_widths[col] + 2)
            print(row)

    print(f"\n[{len(stats_headers)} rows x {len(num_data.columns)} columns]")

    # ----- FIN AFFICHAGE SUR LE TERMINAL -------------------------------------

    # enregistrement dans un dataFrame et exportation pour visualisation
    print(type(stats_data))
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv("describe_stats.csv")
    print("Les statistiques ont été sauvegardées dans 'describe_stats.csv'")


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
        ft_describe(data)
    else:
        print(f"Erreur : le fichier '{filename}' n'a pas été trouvé.")


if __name__ == "__main__":
    main()
