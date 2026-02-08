import numpy as np
import pandas
import pandas as pd
from sklearn.metrics import rand_score
from sklearn.preprocessing import minmax_scale
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import RidgeClassifier
import joblib

class Modell:

    #alle Variablen die innerhalb der Klasse genutzt werden, werden vorbereitet
    def __init__(self):
        self.penguins = pd.read_csv('penguins.csv')
        self.labels = pd.DataFrame()
        self.prep_penguins()
        self.modell = joblib.load("modell.pkl")
        self.penguins_scaled = self.scale_columns(self.vorbereitung(self.penguins))
        self.penguins_new = self.penguins_scaled

    #der initiale Dataframe mit den Palmer Pinguinen wird vorbereitet
    def prep_penguins(self):
        self.penguins.drop(columns='rowid', inplace=True)
        self.penguins.dropna(inplace=True)
        self.labels = pd.DataFrame({'labels': self.penguins['species'].copy()})
        self.penguins.drop(columns='species', inplace=True)
        #replace_dict = {'Torgersen': 0, 'Biscoe': 0.5, 'Dream': 1, 'male': 0, 'female': 1}
        self.vorbereitung(self.penguins)

    #Dataframes werden für die KI Modelle vorbereitet
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

    #Funkltion zum externen Aufruf der predict Methode
    def vorhersage(self, df):
        return self.modell.predict(df)


    #Modell aktualisieren, in dem der neue Dataframe zu den Trainingsdaten hinzugefügt wird.
    def modell_aktualisierung(self, erg, df):
        self.vorbereitung(df)
        self.penguins_new = pd.concat([self.penguins_new, df], ignore_index=True)
        lb = pd.DataFrame.from_dict({"labels": erg})
        self.labels = pd.concat([self.labels, lb] , ignore_index=True)
        ridge = RidgeClassifier()
        self.modell= ridge.fit(self.penguins_new, self.labels['labels'])

    #bestehendes Modell abspeichern
    def modell_speichern(self, name):
        joblib.dump(self.modell, name)

    #neues Modell einstellen
    def modell_wechseln(self, name):
        self.modell = joblib.load(name)

#if __name__ == '__main__':
    #Genauigkeitstest der verschiedenen Modelle
    #M = Modell()
    #X,y = M.penguins, M.labels
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


    #knn=KNeighborsClassifier(algorithm='brute', n_neighbors=5)
    #knn.fit(X_train, y_train['labels'])
    #bernoulliNB = BernoulliNB()
    #bernoulliNB.fit(X_train, y_train['labels'])
    #ridge = RidgeClassifier()
    #ridge.fit(X, y['labels'])

    #def accuracy(modell):
    #    arr = modell.predict(X_test)
    #    korr = 0
    #    falsch = 0
    #    for i in range(len(y_test['labels'])):
    #        if arr[i] == y_test['labels'].iloc[i]:
    #            korr += 1
    #        else:
    #            falsch += 1
    #    return 'korrekt:' + str(korr) +  ' ,falsch:' + str(falsch)

    #print('Genauigkeit KNN')
    #print(accuracy(knn))

    #print('Genauigkeit Naive Bayes')
    #print(accuracy(bernoulliNB))

    #print('Genauigkeit Ridge')
    #print(accuracy(ridge))

