import numpy as np
import pandas
import pandas as pd
from sklearn.preprocessing import minmax_scale
from sklearn.neighbors import KNeighborsClassifier
import joblib

class Modell:

    def __init__(self):
        self.penguins = pd.read_csv('penguins.csv')
        self.labels = pd.DataFrame()
        self.prep_penguins()
        self.modell = joblib.load("modell.pkl")
        self.penguins_scaled = self.scale_columns(self.vorbereitung(self.penguins))
        self.penguins_new = self.penguins_scaled


    def prep_penguins(self):
        self.penguins.drop(columns='rowid', inplace=True)
        self.penguins.dropna(inplace=True)
        self.labels = pd.DataFrame({'labels': self.penguins['species'].copy()})
        self.penguins.drop(columns='species', inplace=True)
        #replace_dict = {'Torgersen': 0, 'Biscoe': 0.5, 'Dream': 1, 'male': 0, 'female': 1}
        self.vorbereitung(self.penguins)

    def vorbereitung(self, df):
        df.replace(to_replace='Torgersen', value=0, inplace=True)
        df.replace(to_replace='Biscoe', value=0.5, inplace=True)
        df.replace(to_replace='Dream', value=1, inplace=True)

        df.replace(to_replace='male', value=0, inplace=True)
        df.replace(to_replace='female', value=1, inplace=True)
        return df

    #Normalisierung
    def scale_columns(self, df):
       for col in df.columns:
            df[col] = minmax_scale(df[col])
            return df


    def vorhersage(self, df):
        return self.modell.predict(df)



    def modell_aktualisierung(self, erg, df):
        self.vorbereitung(df)
        self.penguins_new = pd.concat([self.penguins_new, df], ignore_index=True)
        knn = KNeighborsClassifier(algorithm='brute', n_neighbors=5)
        d = {"labels": erg}
        lb = pd.DataFrame(data=d)
        self.labels = pd.concat([self.labels, lb] , ignore_index=True)
        self.modell= knn.fit(self.penguins_new, self.labels['labels'])


    def modell_speichern(self, name):
        joblib.dump(self.modell, name)

    def modell_wechseln(self, name):
        self.modell = joblib.load(name)
