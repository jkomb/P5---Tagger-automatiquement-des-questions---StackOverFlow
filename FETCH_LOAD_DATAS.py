__doc__ = """Ce module contient la définition des variables de chemins de destination ainsi que l'importation des librairies nécessaires à la définition des 2 fonctions suivantes:
	- fetch_olist_data() : qui sert à télécharger dans un sous-dossier du dossier de travail, 'datasets', l'ensemble des jeux de données nécessaire à notre travail
	- load_olist_data() : qui sert à charger ce jeu de données dans un DataFrame
"""

import os
import pandas as pd

DATA_PATH = "queries_results"

def load_stackoverflow_questions(data_path=DATA_PATH):
    """
        fonction de chargement des données extraites et concaténées
        dans un dataframe

    """
    list_dir = os.listdir(DATA_PATH)
    for file in list_dir:
        file_path = os.path.join(DATA_PATH, file)
        if list_dir.index(file) == 0:
            df_tmp = pd.read_csv(file_path)
        else:
            df_tmp = pd.concat([df_tmp, pd.read_csv(file_path)], axis=0)

    return df_tmp