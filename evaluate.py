"""
Put these files in the same folder as `houses.csv` and `dataset_truth.csv`.

Usage:
    $ python evaluate.py
"""
from __future__ import print_function

import csv
import os.path
import sys


def load_csv(filename):
    """Load a CSV file and return a list with datas.

    (corresponding to truths or predictions).
    """
    datas = list()
    with open(filename, 'r') as opened_csv:
        read_csv = csv.reader(opened_csv, delimiter=',')
        for line in read_csv:
            datas.append(line[1])
    # Clean the header cell
    datas.remove("Hogwarts House")
    return datas

if __name__ == '__main__':
    file = "dataset_truth.csv"
    if os.path.isfile(file):
        truths = load_csv(file)
    else:
        sys.exit("Error: missing dataset_truth.csv in the current directory.")
    if os.path.isfile("houses.csv"):
        predictions = load_csv("houses.csv")
    else:
        sys.exit("Error: missing houses.csv in the current directory.")
    # Here we are comparing each values and counting each time
    count = 0
    # ---------- RAJOUT POUR VOIR QUELLE LIGNE DIVERGE ---------
    divergences = []  # Liste des divergences
    # ---------- RAJOUT POUR VOIR QUELLE LIGNE DIVERGE ---------
    print(len(truths) , len(predictions))
    if len(truths) == len(predictions):
        for i in range(len(truths)):
            if truths[i] == predictions[i]:
                count += 1
    # ---------- RAJOUT POUR VOIR QUELLE LIGNE DIVERGE ---------
            else:
                divergences.append((i, truths[i], predictions[i]))
    score = float(count) / len(truths)
    print("Your score on test set: %.3f" % score)
    if score >= .98:
        print("Good job! Mc Gonagall congratulates you.")
    else:
        print("Too bad, Mc Gonagall flunked you.")

    # ----------- AFFICHE LES LIGNES DIVERGENTES ---------------
    if divergences:
        print("\nLines with discrepancies:")
        for index, truth, prediction in divergences:
            print(f"Line {index + 1}: Truth = {truth}, Prediction = {prediction}")
    else:
        print("All lines match perfectly!")

