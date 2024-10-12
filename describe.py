"""Describe program.

Ce programme ouvre un fichier de données compressées au format .tgz et
les mets en forme suivant plusieurs types de visualisations graphiques.
"""

import os.path
import sys
import tarfile

import pandas as pd


def moyenne(val: pd.Series, count: float) -> float:
    if not val.empty:
        mean = sum(val) / count
        return round(mean, 6)
    return None


def maximum(val: pd.Series) -> float:
    if not val.empty: # s'il y a des valeurs non-nulles
        tmp_max = val.iloc[0] # on initialise avec la première valeur
        for value in val.iloc[1:]:
            if value > tmp_max:
                tmp_max = value
        return round(tmp_max, 6)
    return None # S'il n'y a pas de valeurs non-nulles


def minimum(val: pd.Series) -> float:
    if not val.empty: # s'il y a des valeurs non-nulles
        tmp_min = val.iloc[0] # on initialise avec la première valeur
        for value in val.iloc[1:]:
            if value < tmp_min:
                tmp_min = value
        return round(tmp_min, 6)
    return None # S'il n'y a pas de valeurs non-nulles


def ft_describe(data: pd.DataFrame) -> None:
    """Reproduit la fonction describe de python.

    Cette fonction prend en paramètre un dataframe pandas, et en extrait :
    :-> count (num + string + time) -> Count number of non-NA/null observations
    :unique(string + time)
    :top (string + time) -> most common value
    :freq (string + time) -> most common value's frequency
    :first (time) -> the first item
    :last (time) -> the last item
    :-> max (num) -> Maximum of the values in the object
    :-> min (num) -> Minimum of the values in the object
    :-> mean (num) -> Mean of the values
    :-> std (num) -> Standard deviation of the observations (= écart-type)
    :-> 25 (num) -> lower percentile
    :-> 50 (num) -> median percentile
    :-> 75 (num) -> upper percentile
    :select_dtypes -> Subset including/excluding columns based on their dtype
    ATTENTION for mixed data types in a dataFrame, the default is to return
    only an analysis of numeric columns !!! (SUJET = NUMERICAL FEATURES ONLY !)
    """
    # On sélectionne uniquement les colonnes numériques
    num_data = data.select_dtypes(include=['number'])

# -----------
    print(num_data.describe(include='all'))
    test = num_data[~num_data["Charms"].isna()][num_data.columns[1:7]]
    print(test)
    test = num_data[~num_data["Charms"].isna()][num_data.columns[8:]]
    print(test)
    print(data.dtypes)
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

    # 2) MIN
    min_values = {}
    max_values = {}
    mean_values = {}
    for col in num_data.columns:
        # Exclure les valeurs nulles pour le calcul du min
        non_null_values = num_data[~num_data[col].isna()][col]
        # non_null_values = [value for value in num_data[column] if not pd.isnull(value)]
        min_values[col] = minimum(non_null_values)
        max_values[col] = maximum(non_null_values)
        mean_values[col] = moyenne(non_null_values, count_values[col])

    # 5) STD -> Standard deviation (= écart-type)
    std_values = {}
    for column in num_data.columns:
        # Exclure les valeurs nulles pour le calcul du max
        non_null_values = [value for value in num_data[column] if not pd.isnull(value)]
        if non_null_values:
            mean = mean_values[column]
            
        else:
            std_values[column] = None

    # Ajoutez d'autres statistiques ici
    stats_headers = ['count', 'min', 'max', 'mean']

    # Déterminer la largeur max de la colonne d'index
    index_col_width = max(len(col) for col in num_data.columns) + 2

    # Imprimer les en-tête de colonne (statistiques : count, min, etc...)
    header = f"{'':<{index_col_width}}" # Colonne d'index (statistiques)
    for stat in stats_headers:
        header += f"{stat:<15}"
    print(header)

    # Imprimer chaque colonne du DataFrame avec ses statistiques
    for col in num_data.columns:
        row = f"{col:<{index_col_width}}" # Ajustement de la colonne d'index
        row += f"{count_values[col]:<15}"
        if min_values[col] is not None:
            row += f"{min_values[col]:<15}"
        else:
            row += f"{'N/A':<15}"
        if max_values[col] is not None:
            row += f"{max_values[col]:<15}"
        else:
            row += f"{'N/A':<15}"
        if mean_values[col] is not None:
            row += f"{mean_values[col]:<15}"
        else:
            row += f"{'N/A':<15}"
        print(row)


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
    try :
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
        # with open(file_path, 'r') as f:
        #     data = f.read()
        #     print("contenu du fichier : \n")
        #     print(data)
        data = load(file_path)
        if data is None:
            sys.exit(1)
        ft_describe(data)
    else:
        print(f"Erreur : le fichier '{filename}' n'a pas été trouvé.")


if __name__ == "__main__":
        main()
