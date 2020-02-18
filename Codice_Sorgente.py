import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

data = pd.read_csv("Matriz_Completa.csv")

# La classe StratifiedShuffleSplit esegue uno split del dataset in train_set e test_set rispettando la
# distribuzione dei valori di tutto il dataset. In questo caso stiamo stratificando rispetto alla colonna TIPO.
# L'attributo test_size = 0.2 indica che il test_set comprende il 20% dei valori dell'intero dataset

split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
for train_index, test_index in split.split(data, data["TIPO"]):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]

# Qui abbiamo le percentuali di siti Benigni e Maligni nel dataset
data["TIPO"].value_counts()/len(data)

# Qui abbiamo le dimensioni di test_set e train_set
print(len(strat_train_set), len(strat_test_set))

# Come possiamo vedere, le percentuali sono state rispettate
strat_train_set["TIPO"].value_counts()/len(strat_train_set)

def getValidIndex(invalidIndex, lenght):
    indici = np.array(range(lenght))
    valid = np.delete(indici, invalidIndex)
    return valid

class Column_selector(BaseEstimator, TransformerMixin):
    def __init__(self, col_indexes):
        self.col_indexes = col_indexes
    def fit(self, X, y= None):
        return self
    def transform(self, X):
        return X[:, self.col_indexes].reshape(len(X[:, self.col_indexes]), len(self.col_indexes))
    
class StringUpper(BaseEstimator, TransformerMixin):
    def __init__(self):
        return None
    def fit(self, X, y= None):
        return self
    def transform(self, X):
        r = [str(element).upper() for element in X]
        return np.array(r).reshape(-1, 1)
    
class ColumnLabelBinarizer(LabelBinarizer):
    def fit_transform(self,X,y=None):
        return super(ColumnLabelBinarizer,self).fit_transform(X)
    
class NumberCheckNan(BaseEstimator, TransformerMixin):
    def __init__(self):
        return None 
    def fit(self, X, y= None):
        return self
    def transform(self, X):
        for i in range (0, len(X)):
            for j in range(0, len(X[0])):
                if X[i][j] is None or X[i][j]!= X[i][j] or X[i][j] == "None":
                    X[i][j] = 0
        return np.array(X).reshape(-1, len(X[0]))
    
    
class CheckNan(BaseEstimator, TransformerMixin):
    def __init__(self):
        return None 
    def fit(self, X, y= None):
        return self
    def transform(self, X):
        for i in range (0, len(X)):
            if X[i]!= X[i]:
                X[i] = "None"
        return np.array(X).reshape(-1, 1)
    
class DataFrameToArray(BaseEstimator, TransformerMixin):
    def __init__(self):
        return None  
    def fit(self, X, y= None):
        return self
    def transform(self, X):        
        return X.values

    
class ApacheTransform(BaseEstimator, TransformerMixin):
    #col_indexes Indici sui quali la pipeline deve operare
    def __init__(self):
        return None  
    def fit(self, X, y= None):
        return self
    def transform(self, X):
        #print(X)
        #X è t1 passato come parametro in c= Server_Pipeline.fit_transform(t1)
        Y=np.array([])
        for i in range(0, len(X)):
            Y= np.append(Y,int("APACHE" in X[i]))
        return Y.reshape(-1, 1)

class CacheHandler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.value_list = {}
        return None
    def fit(self, X, y= None):
        X = X.tolist()
        for j in range (0, len(X)):
            value = X[j]
            if X[j] is None:
                cell_values="None"
            else:
                cell_values= "".join(X[j][0].split(" ")).split(";")
            for cell_value in cell_values:
                if cell_value not in self.value_list:
                    self.value_list[cell_value] = len(self.value_list.keys())
        return  self  
    def transform(self, X):
        matrix=[]
        for i in range (0, len(X)):
            cell_values=""
            vettoreIesimaCella= (np.zeros(len(self.value_list.keys()),dtype=int)).tolist()
            if X[i] is None:
                cell_values="None"
            else:
                cell_values= "".join(X[i][0].split(" ")).split(";")
            for cell_value in cell_values:
                if cell_value in self.value_list:
                    indexToChange=self.value_list[cell_value]
                    vettoreIesimaCella[indexToChange]=1 
            matrix.append(vettoreIesimaCella) 
        return np.array(matrix).reshape(-1, len(matrix[0]))
    
class extension_extractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        return None
    def fit(self, X, y= None):
        return self
    def transform(self, X):
        result = []
        for i in range(len(X)):
            substrings = X[i][0].split(".") 
            result.append(substrings[len(substrings)-1])
        return np.array(result)

prePipeline = Pipeline([
    ('dataFrameSelector', DataFrameToArray()),
    ('column_selector', Column_selector(col_indexes=getValidIndex([0,9,10], len(strat_train_set.columns))))
])

charsetPipeline = Pipeline([
    ('Column_selector', Column_selector(col_indexes=[2])),
    ('checkNan', CheckNan()),
    ('stringUpper', StringUpper()),
    ('columnLabelBinarizer',ColumnLabelBinarizer())
])

Server_Apache_Pipeline = Pipeline([
     ("column_selector", Column_selector(col_indexes=[3])),
     ("CheckNan", CheckNan()),
     ("StringUpper", StringUpper()),
     ("ApacheTransform", ApacheTransform())
   ])

Server_Pipeline = Pipeline([
    ("column_selector", Column_selector(col_indexes=[3])),
     #Effettuo l'UPPERCASE della colonna selezionata e la passo al transform successivo
    ("check_nan", CheckNan()),
    ("StringUpper", StringUpper()),
    #Prendo l'UPPERCASE e trasformo tutto in binario
    ("ColumnLabelBinarizer", ColumnLabelBinarizer())
])

server_feature_union = FeatureUnion([
    ("server_pipeline_1", Server_Apache_Pipeline),
    ("Server_Pipeline", Server_Pipeline)
])

Cache_Pipeline = Pipeline([
    ("column_selector", Column_selector(col_indexes=[4])),
    ("StringUpper", StringUpper()), 
    ("CacheHandler", CacheHandler())
])

content_length_pipeline = Pipeline([
    ('column_selector', Column_selector(col_indexes = [5])),
    ('imputer', Imputer(strategy = "median")),
    ('std_scaler', StandardScaler())
])

countryPipeline = Pipeline([
    ('column_selector', Column_selector(col_indexes=[6])),
    ('checkNan', CheckNan()),
    ('stringUpper', StringUpper()),
    ('columnLabelBinarizer',ColumnLabelBinarizer())
])

provincePipeline = Pipeline([
    ('column_selector', Column_selector(col_indexes=[7])),
    ('checkNan', CheckNan()),
    ('stringUpper', StringUpper()),
    ('columnLabelBinarizer',ColumnLabelBinarizer())
])

portPipeline = Pipeline([
    ('column_selector', Column_selector(col_indexes=[10])),
    ('checkNan', CheckNan()),
    ('stringUpper', StringUpper()),
    ('columnLabelBinarizer',ColumnLabelBinarizer())  
])

domain_pipeline = Pipeline([
    ("column_selector", Column_selector([8])),
    ("extension_extractor", extension_extractor()),
    ("label_binarizer", ColumnLabelBinarizer())
])

select_and_scale1 = Pipeline([
    ("column_selector", Column_selector([0, 1])),
    ("checkNumberNan", NumberCheckNan()),
    ('std_scaler', StandardScaler())
])

select_and_scale2 = Pipeline([
    ("column_selector", Column_selector([9])),
    ("checkNumberNan", NumberCheckNan()),
    ('std_scaler', StandardScaler())
])

select_and_scale3 = Pipeline([
    ("column_selector", Column_selector([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])),
    ("checkNumberNan", NumberCheckNan()),
    ('std_scaler', StandardScaler())
])

full_pipeline = FeatureUnion([
    #("add_url_length", Column_selector([0])),
    #("add_num_special_charater", Column_selector([1])),
    ('select_and_scale1', select_and_scale1),
    ("charset_pipeline", charsetPipeline),
    ("server_pipeline", server_feature_union),
    ("cache_pipeline", Cache_Pipeline),
    ("content_length_pipeline", content_length_pipeline),
    ("countryPipeline", countryPipeline),
    ("provincePipeline", provincePipeline),
    ("domain_pipeline", domain_pipeline),
    #("add_col_9", Column_selector([9])),
    ('select_and_scale2', select_and_scale2),
    ("portPipeline", portPipeline),
    ('select_and_scale3', select_and_scale3)
    #("add_remaining_cols_", Column_selector([11, 12, 13, 14, 15, 16, 17, 18, 19, 20]))
])

# Definisco una pipeline che mette insieme la prePipeline e la full_pipeline, in modo da non dover fare due invocazioni

data_preparator = Pipeline([
    ("prePipeline", prePipeline),
    ("full_pipeline", full_pipeline)
])

# Preparo i dati per gli algoritmi
train_set_prepared = data_preparator.fit_transform(strat_train_set)

# Definisco un label_binarizer, questo servirà a trasformare in binario le etichette 'Benigno'/'Maligno'
label_binarizer = LabelBinarizer()

# Qui possiamo vedere la struttura del train_set trasformato
train_set_prepared.shape

# Otteniamo le etichette binarizzate per il train_set. Notare che qui viene invocato il metodo .fit_transform() !!
train_set_labels = label_binarizer.fit_transform(strat_train_set["TIPO"])

# Usiamo la funzione flatten per modificare la formattazione di serie del vettore che è
# del tipo v = [[e1], [e2], [e3], ..., [en]] in v = [e1, e2, e3, ..., en]
train_set_labels = train_set_labels.flatten()

# Definiamo una funzione che misura l'accuratezza delle predizioni dell'algoritmo
def accuracy_rate(labels, predictions):
    count = 0
    for i in range(len(labels)):
        if labels[i] == predictions[i]:
            count += 1
    return count/len(labels)

# Importiamo il modello SVC. Esso è un Classificatore binario (riesce a classificare tra due sole classi) basato
# sull'algoritmo SVM = Support Vector Machines.
from sklearn.svm import LinearSVC

# Facciamo un test preventivo per misurare l'accuratezza dell'algoritmo, lasciando i parametri di default
classifier = LinearSVC(C= 1, loss = "hinge")

# Eseguiamo il fit (= allenamento) dell'algoritmo sul train_set
classifier.fit(train_set_prepared, train_set_labels)

# Misuriamo l'accuratezza del classificatore sul train_set, ovvero sui dati sul quale è stato allenato
predictions_train = classifier.predict(train_set_prepared)

# Misuriamo l'accuratezza delle predizioni
accuracy_rate(train_set_labels, predictions_train)

# L'accuratezza sembra eccessiva, facendo pensare che l'algoritmo sia andato in overfitting. Per regolarizzare
# le predizioni, usiamo la classe GridSearchCV, la quale divide il train_set in tante coppie di 
# mini_train_set/mini_test_set ed allena l'algoritmo con vari parametri, restituendoci quelli che
# hanno performato meglio
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'C': [.1, .5, 1, 10, 50, 100, 500]}, 
    {"loss": ["hinge", "squared_hinge"]}
]
classifier = LinearSVC()

# Alleniamo l'algoritmo nel modo prestabilito. Il parametri cv indica il numero di split del train_set in
# 'mini_train_set'/'mini_test_set', e quindi le dimensioni degli stessi.
grid_search = GridSearchCV(classifier, param_grid, cv = 5, scoring = 'neg_mean_squared_error')
grid_search.fit(train_set_prepared, train_set_labels)

# Chiediamo semplicemente a grid_search qual è la migliore combinazione possibile di parametri: il fatto che
# la misura dell'errore non sia specificata ci indica che è indifferente ai fini delle performances
grid_search.best_params_

#Facciamo una ricerca più 'raffinata' per il parametro 'C': il risultato ci indica che C=1 ci assicura
#La maggiore accuratezza
grid_search = GridSearchCV(classifier, param_grid= [{'C': [7, 8.5, 1, 11.5, 13]}], cv = 5, scoring = 'neg_mean_squared_error')
grid_search.fit(train_set_prepared, train_set_labels)
grid_search.best_params_

#Siamo pronti quindi per misurare le performances sul test_set! Innanzitutto lo alleniamo con i parametri migliori, e
#ne misuriamo nuovamente l'accuratezza sul train_set
classifier = LinearSVC(C= 1, loss = "hinge")
classifier.fit(train_set_prepared, train_set_labels)
predizioni = classifier.predict(train_set_prepared)
accuracy_rate(train_set_labels, predizioni)

# A questo punto calcoliamo le predizioni sul test_set (dati che l'algoritmo non ha mai visto), e ne misuriamo 
# l'accuratezza.
test_set_prepared = data_preparator.transform(strat_test_set)
test_set_labels = label_binarizer.transform(strat_test_set["TIPO"])
predizioni_test = classifier.predict(test_set_prepared)
accuracy_rate(test_set_labels, predizioni_test)

# Possiamo vedere che l'accuratezza sul test_set (= dati che l'algoritmo non ha mai visto) raggiunge il 98.6%

# Alleniamo adesso un classificatore basato su reti neurali trmite tensorflow
import tensorflow as tf

# Creiamo un oggetto iterabile sulle colonne del test_set
feature_cols = tf.contrib.learn.infer_real_valued_columns_from_input(train_set_prepared)

# Creiamo una istanza di DNNClassifier, un classificatore basato su reti neurali profonde. In questo caso abbiamo tre
#'livelli nascosti', che si compongono rispettivamente di 100, 70 e 50 neuroni. Abbiamo scelto di usare
# una rete a tre livelli (ma con un numero moderato di neuroni) piuttosto che avere 1-2 livelli ma più ampi,
# poiché aumentare la profondità della rete assicura performances migliori rispetto ad aumentarne l'ampiezza.
dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[100, 70, 50], n_classes = 2, feature_columns = feature_cols)

# Questo cast è necessario per le versioni più recenti di TesorFlow
dnn_clf = tf.contrib.learn.SKCompat(dnn_clf)

# Eseguiamo il training dell'algoritmo. Con questi parametri l'algoritmo ripete 500 iterazioni del training e,
# ad ogni iterazione, considera 40 elementi del train_set (scelti uniformemente a caso) 
dnn_clf.fit(train_set_prepared, train_set_labels, batch_size = 40, steps=300)

# Calcoliamo le predizioni riguardo al train_set
predizioni_train = dnn_clf.predict(train_set_prepared)

# Calcoliamo l'accuratezza delle precisioni
accuracy_rate(train_set_labels, predizioni_train["classes"])

# L'accuratezza sul test_set risulta molto alta: ciò può essere dovuto sia alla eccessiva complessità del modello,
# che può aver appreso anche il 'rumore' presente sui dati, andando in overfitting, sia al fatto che il modello
# risulti effettivamente accurato. Pur essendo l'accuratezza così alta, essa potrebbe essere comunque indice di
# un modello accurato siccome questo numero non si discosta tanto dalla accuratezza ottenuta tramite
# il classificatore SVM. Perciò, ci limiteremo a controllare se il modello è affidabile o meno, semplicemente
# controllando l'accuratezza sul test_set.
predicted_test = dnn_clf.predict(test_set_prepared)

accuracy_rate(test_set_labels.flatten(), predicted_test["classes"])

# Proviamo inoltre ad allenare un ultimo modello, detto RandomForestClassifier. Esso è basato sull'algoritmo 
# DecisionTree, e consiste nel creare n_estimators Decision Trees, ognuna con un numero massimo di nodi
# uguale a max_leaf_nodes. In questo primo caso utilizziamo dei parametri di esempio, con lo scopo di affinarli
# successivamente

from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(n_estimators = 100, max_leaf_nodes = 13)

# Alleniamo l'algoritmo tramite il metodo .fit()
rnd_clf.fit(train_set_prepared, train_set_labels)

#Misuriamo l'accuratezza delle predizioni sui dati sui quali l'algoritmo è stato allenato.
predizioni_train = rnd_clf.predict(train_set_prepared)
accuracy_rate(predizioni_train, train_set_labels)

# Utilizziamo la classe grid_search per fare diversi test sugli iperparametri dell'algoritmo
param_grid = [
    {"max_leaf_nodes": [5, 8, 10, 12, 15]},
    {'n_estimators': [50, 75, 100, 175, 250, 500]}
]
grid_search = GridSearchCV(rnd_clf, param_grid, cv = 5, scoring = 'neg_mean_squared_error')
grid_search.fit(train_set_prepared, train_set_labels)

# Eseguiamo quindi una ricerca più mirata sul parametro max_leaf_nodes
param_grid = [
    {"max_leaf_nodes": [13, 14, 15, 16, 17, 18, 19]},
    {'n_estimators': [50]}
]
grid_search = GridSearchCV(rnd_clf, param_grid, cv = 5, scoring = 'neg_mean_squared_error')
grid_search.fit(train_set_prepared, train_set_labels)
grid_search.best_params_

# Ancora...
param_grid = [
    {"max_leaf_nodes": [19, 20, 21, 22, 23, 24, 25, 26]},
    {'n_estimators': [50]}
]
grid_search = GridSearchCV(rnd_clf, param_grid, cv = 5, scoring = 'neg_mean_squared_error')
grid_search.fit(train_set_prepared, train_set_labels)
grid_search.best_params_

# eseguiamo un training dell'algoritmo con i parametri indicati da grid_search, e ne misuriamo l'accuratezza
rnd_clf = RandomForestClassifier(n_estimators = 50, max_leaf_nodes = 24)
rnd_clf.fit(train_set_prepared, train_set_labels)
predizioni_train = rnd_clf.predict(train_set_prepared)
accuracy_rate(predizioni_train, train_set_labels)

# Testiamo l'algoritmo sul test_set
predizioni_test = rnd_clf.predict(test_set_prepared)
accuracy_rate(predizioni_test, test_set_labels)