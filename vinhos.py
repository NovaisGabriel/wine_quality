
# coding: utf-8

# # Teste Cognitivo A.I.
# #### Solução : Gabriel Novais

# ### Importando os Pacotes

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras 
import tensorflow
import theano
import sys
import seaborn as sns
import warnings

from sklearn.metrics import fbeta_score, accuracy_score
from sklearn.model_selection import train_test_split,cross_val_score,RandomizedSearchCV,GridSearchCV
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV

get_ipython().run_line_magic('load_ext', 'autotime')
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings("ignore")


# ### Função Medida

def medida(modelo, X_test, y_test):
    pred = modelo.predict(X_test)
    errors = abs(pred - y_test)**2
    media = np.mean(errors)
    mape = 100 * np.mean(errors / y_test)
    acc = 100 - mape
    print('MSE: {:0.2f}%.'.format(media))
    print('Accuracy = {:0.2f}%.'.format(acc))
    
    return acc,media


# ### Importando os Dados

vinhos = pd.read_csv("winequality.csv",delimiter=';',dtype=str)
vinhos.head()


# ### Corrigindo Erros nos Dados


#Alcohol apresentou erro de preechimento
for column in vinhos.columns[1:]:
    vinhos[column] = vinhos[column].apply(lambda x: float(x) if len(x.split('.'))<3 else 'ERRO')
perdas = vinhos[vinhos['alcohol']== 'ERRO'].shape[0]
vinhos = vinhos[vinhos['alcohol']!= 'ERRO'].reset_index(drop=True)
vinhos['alcohol'] = vinhos['alcohol'].astype(float)
print("Total de erros em alcohol : "+str(perdas))
print("Dados Corrigidos")


# ### Análise exploratória dos Dados

vinhos_obs = vinhos

#Transformando type em dummy (White = 1, Red = 0):
labelencoder= LabelEncoder()
vinhos_obs['type']=labelencoder.fit_transform(vinhos_obs['type'])
vinhos_obs.head()

vinhos_obs.groupby('type').count()


vinhos.describe()


hist_Type = vinhos.hist('type')
hist_FixedAcidity = vinhos.hist('fixed acidity')
hist_VolatileAcidity = vinhos.hist('volatile acidity')
hist_CitricAcid = vinhos.hist('citric acid')
hist_ResidualSugar = vinhos.hist('residual sugar')
hist_Chlorides = vinhos.hist('chlorides')
hist_FreeSulfurDioxide = vinhos.hist('free sulfur dioxide')
hist_TotalSulfurDioxide = vinhos.hist('total sulfur dioxide')
hist_Density = vinhos.hist('density')
hist_Ph = vinhos.hist('pH')
hist_Sulphates = vinhos.hist('sulphates')
hist_Alcohol = vinhos.hist('alcohol')
hist_Quality = vinhos.hist('quality')

sns.pairplot(vinhos_obs)
plt.show()


Var_Corr = vinhos_obs.corr()
fig,ax = plt.subplots(figsize = (10,10))
sns.heatmap(Var_Corr, xticklabels=Var_Corr.columns, yticklabels=Var_Corr.columns, annot=True, ax = ax)
plt.show()


# ### Preparando os Dados (ANN)

X = vinhos_obs.iloc[:,1:-1].values
y = vinhos_obs.iloc[:,-1].values

max1 = max(X[:,0])
onehotencoder_X = OneHotEncoder(categorical_features = [0], n_values= [max1+1])
X = onehotencoder_X.fit_transform(X).toarray()
onehotencoder_y = OneHotEncoder(categorical_features = [0])
y = onehotencoder_y.fit_transform(y.reshape((-1,1))).toarray()
X = pd.DataFrame(X)
X = X.drop([0], axis = 1)
X = X.iloc[:,:].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print('Total de observações : '+str(len(X)))
print('Total de observações do treinamento : ' + str(len(X_train)))
print('Total de observações do teste : ' + str(len(X_test)))


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# ### Método 1 : Artificial Neural Networks (ANN)

neurons  = round((X.shape[1] + y.shape[1])/2)
n_layers = 10
classifier = Sequential()
classifier.add(Dense(units = neurons,kernel_initializer = 'uniform',activation = 'relu',input_shape = (X.shape[1],)))
for layer in range(n_layers):
    classifier.add(Dense(units = neurons,kernel_initializer = 'uniform',activation = 'relu'))
classifier.add(Dense(units=y.shape[1],kernel_initializer='uniform',activation='softmax'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)


def build_classifier():
    neurons  = round((X.shape[1] + y.shape[1])/2)
    n_layers = 12
    classifier = Sequential()
    classifier.add(Dense(units = neurons,kernel_initializer = 'uniform',activation = 'relu',input_shape = (X.shape[1],)))
    for layer in range(n_layers):
        classifier.add(Dense(units = neurons,kernel_initializer = 'uniform',activation = 'relu'))
    classifier.add(Dense(units=y.shape[1],kernel_initializer='uniform',activation='softmax'))
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    return classifier



classifier  = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)


accuracies  = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = 1)


mean = accuracies.mean()
std  = accuracies.std()
mean, 2*std
print("Mean : " + str("{0:.2f}".format(mean*100))+"%")
print("Variance : " + str("{0:.2f}".format(std*100))+"%")


# ### Preparando os dados (GNB, MLR, RTC, DTC)

X = vinhos_obs.iloc[:,1:-1].values
y = vinhos_obs.iloc[:,-1].values


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print('Total de observações : '+str(len(X)))
print('Total de observações do treinamento : ' + str(len(X_train)))
print('Total de observações do teste : ' + str(len(X_test)))


# ### Método 2 : Gaussian Naive Bayes (GNB) 

param_grid_GNB = {
    'priors': [None],
    'var_smoothing': [1e-09,1e-05]
}


GNB = GaussianNB()


grid_search_GNB = GridSearchCV(estimator = GNB, param_grid = param_grid_GNB, cv = 10, verbose = 2)


grid_search_GNB.fit(X_train, y_train)
best_params = grid_search_GNB.best_params_
best_grid_GNB = grid_search_GNB.best_estimator_


grid_accuracy_GNB = medida(best_grid_GNB, X_test, y_test)


# ### Método 3 - Multinomial Logistic Regression (MLR)


MLR = LogisticRegressionCV(cv=3,multi_class='multinomial', solver='newton-cg')


MLR.fit(X_train, y_train)


grid_accuracy_MLR = medida(MLR, X_test, y_test)


# ### Método 4 - Random Forest Classifier (RFC)

param_grid_RFC = {
    'bootstrap': [True],
    'max_depth': [40,80],
    'max_features': [2,10],
    'min_samples_leaf': [3,8],
    'min_samples_split': [2,8],
    'n_estimators': [50]
}

RFC = RandomForestClassifier()

grid_search_RFC = GridSearchCV(estimator = RFC, param_grid = param_grid_RFC, cv = 10, verbose = 2)

grid_search_RFC.fit(X_train, y_train)
grid_search_RFC.best_params_
best_grid_RFC = grid_search_RFC.best_estimator_


grid_accuracy_RFC = medida(best_grid_RFC, X_test, y_test)


# ### Método 5 - Decision Tree Classifier (DTC)

param_grid_DTC = {
    'max_depth': [40,100],
    'max_features': [2,5],
    'min_samples_leaf': [3,8],
    'min_samples_split': [2,8],
    'random_state':[None,0]
}


DTC = DecisionTreeClassifier()


grid_search_DTC = GridSearchCV(estimator = DTC, param_grid = param_grid_DTC, cv = 10, verbose = 2)


grid_search_DTC.fit(X_train, y_train)
grid_search_DTC.best_params_
best_grid_DTC = grid_search_DTC.best_estimator_


grid_accuracy_DTC = medida(best_grid_DTC, X_test, y_test)


# ### Comparação entre os métodos

acc_ANN, acc_MLR, acc_GNB, acc_RFC, acc_DTC = mean*100, grid_accuracy_MLR[0], grid_accuracy_GNB[0], grid_accuracy_RFC[0], grid_accuracy_DTC[0]


mse_ANN, mse_MLR, mse_GNB, mse_RFC, mse_DTC = std, grid_accuracy_MLR[1], grid_accuracy_GNB[1], grid_accuracy_RFC[1], grid_accuracy_DTC[1]

resultados = pd.DataFrame({'Método':['ANN','GNB','MLR','RFC','DTC'],
                           'Accuracy':[acc_ANN,acc_GNB,acc_MLR,acc_RFC,acc_DTC],
                           'MSE':[mse_ANN,mse_GNB,mse_MLR,mse_RFC,mse_DTC]})
resultados.sort_values('MSE')

