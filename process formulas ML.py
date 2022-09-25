# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import KFold, GridSearchCV
import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, matthews_corrcoef 
import sklearn
from sklearn import svm, ensemble
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from matplotlib import pyplot as plt
from  sklearn.neural_network import MLPClassifier
from sklearn import svm
import csv

param_grid_mlp = {
          'activation': ['relu','tanh','logistic'],
          'alpha': [0.0001, 0.05],
          'learning_rate': ['constant','adaptive'],
          'solver': ['adam']}
param_grid_svc= {'C': [0.1, 1, 10 ], 'kernel': ['rbf', 'poly', 'sigmoid'],'degree':[2,3,4], 'gamma': ['auto', 'scale']}

#rs = ensemble.RandomForestClassifier(n_estimators = 20)
#rs =  MLPClassifier(hidden_layer_sizes=50, max_iter=5000)
#rs = KNeighborsClassifier(n_neighbors=1, n_jobs = 5, weights = 'distance' )
rs = svm.SVC(C=0.1, kernel='poly')
training_size = 0
increment = 10


#ns = GridSearchCV(rs, param_grid = param_grid_svc ,error_score=np.nan, cv=cv)
f1 = make_scorer(f1_score, pos_label="True")
mcc = make_scorer(matthews_corrcoef)
accuracy = make_scorer(accuracy_score)
scores = {"f1":f1,"mcc":mcc, "accuracy":accuracy}


train_names = []
train_labels = []
test_names = []
test_labels = []
test_predicted = []

exps = ["12_500","20_500","30_500"]

def get_rar_dataset(filename, n=None):
    if filename ==  "30_500_1000.csv":
        with open(filename, encoding='utf16') as data_file:
            reader = csv.reader(data_file)
            names = np.array(list(next(reader)))
        data = pd.read_csv(filename, dtype=object, encoding="UTF-16")
        data = data.to_numpy()

        n = len(names) - 1

        # ## Extract data names, membership values and Gram matrix

        names = names[1:n+1]
        mu = np.array([(row[0]) for row in data[0:n+1]])
        gram = np.array([[float(k.replace('NA', '0')) for k in row[1:n+1]]
                         for row in data[0:n+1]])

        assert(len(names.shape) == 1)
        assert(len(mu.shape) == 1)
        assert(len(gram.shape) == 2)

        assert(names.shape[0] == gram.shape[0] == gram.shape[1] == mu.shape[0])

        X = np.array([[x] for x in np.arange(n)])

        return X, gram, mu, names
    else:
        with open(filename) as data_file:
            reader = csv.reader(data_file)
            names = np.array(list(next(reader)))

        data = pd.read_csv(filename, dtype=object)
        data = data.to_numpy()
    
        n = len(names) - 1
    
        # ## Extract data names, membership values and Gram matrix
    
        names = names[1:n+1]
        mu = np.array([(row[0]) for row in data[0:n+1]])
        gram = np.array([[float(k.replace('NA', '0')) for k in row[1:n+1]]
                         for row in data[0:n+1]])
    
        assert(len(names.shape) == 1)
        assert(len(mu.shape) == 1)
        assert(len(gram.shape) == 2)
    
        assert(names.shape[0] == gram.shape[0] == gram.shape[1] == mu.shape[0])
    
        X = np.array([[x] for x in np.arange(n)])
    
        return X, gram, mu, names
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
for exp in exps:
    training_size = training_size + increment
    folds= 40 - training_size
    cv = StratifiedShuffleSplit(n_splits=folds, train_size=training_size)
    file_names = [exp + '_base',exp + '_30',exp + '_100',exp + '_1000']
    
    for file_name in file_names:
        X, gram, mu, names = get_rar_dataset(file_name+".csv")
        print('#############################################################################################################')
        print("set: ",file_name)
        X_train, X_test, mu_train, mu_test = train_test_split(X, mu, test_size=500-training_size, stratify=mu)
        #ns.fit(gram, mu)
        #print(ns.best_score_, ns.best_params_, ns.best_estimator_)
        crossval = cross_validate(rs, gram[X.flatten()][:, X.flatten()], mu, cv=cv, scoring =scores)
        #predict_cv = cv.get_n_splits(gram[X.flatten()][:, X.flatten()], mu)
        
        predicted_targets = np.array([])
        actual_targets = np.array([])
        for X_train, X_test in cv.split(X, mu):
            train_x, train_y, test_x, test_y = gram[X_train.flatten()][:, X_train.flatten()], mu[X_train.flatten()], gram[X_test.flatten()][:, X_train.flatten()], mu[X_test.flatten()]
            classifier = rs.fit(train_x, train_y)
            predicted_labels = classifier.predict(test_x)
            predicted_targets = np.append(predicted_targets, predicted_labels)
            actual_targets = np.append(actual_targets, test_y)
            train_names = np.array(names[X_train.flatten()])
            train_labels = np.array(mu[X_train.flatten()])
            test_names = np.array(names[X_test.flatten()])
            test_labels = np.array(mu[X_test.flatten()])
            test_predicted = predicted_labels
        cnf_matrix = confusion_matrix(actual_targets, predicted_targets,labels=['False','True'])
        cnf_matrix_norm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        cnf_matrix_not_norm = cnf_matrix.astype('int') //training_size
        np.set_printoptions(precision=2)
        
        
        # disp= ConfusionMatrixDisplay(cnf_matrix_not_norm)
        # disp.plot(colorbar=False)
        # for labels in disp.text_.ravel():
        #     labels.set_fontsize(30)
        # plt.title(file_name)
        # plt.show()
        
        disp= ConfusionMatrixDisplay(cnf_matrix_norm, display_labels=['False','True'])
        disp.plot(colorbar=False)
        for labels in disp.text_.ravel():
            labels.set_fontsize(30)
        #plt.title(file_name+"_norm")
        plt.show()
        
        #print(crossval)
        print("%0.2f F1 %0.2f MCC with accuracy %0.2f" % (crossval.get("test_f1").mean(), crossval.get("test_mcc").mean(), crossval.get("test_accuracy").mean()))
    print('#############################################################################################################')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

sample_train_set = np.vstack((train_names,train_labels)).T
sample_test_set = np.vstack((test_names,test_labels, test_predicted)).T
np.savetxt("sample_training_formulas.csv", 
           sample_train_set,
           delimiter =", ", 
           fmt ='% s',
           encoding = "UTF-16")
print(sample_train_set)
np.savetxt("sample_test_formulas.csv", 
           sample_test_set,
           delimiter =", ", 
           fmt ='% s',
           encoding = "UTF-16")


#########################################################################################################################################


# file_name=exp + '_30'

# X, gram, mu, names = get_rar_dataset(file_name+".csv")
# print('#############################################################################################################')


# print("set: ",file_name)

# for i in range(1):

#      X_train, X_test, mu_train, mu_test = train_test_split(X, mu, test_size=500-training_size, stratify=mu)

#      train_test = gram[X_train.flatten()][:, X_train.flatten()]
#      test_test = gram[X_test.flatten()][:, X_train.flatten()]
#      test_names = names[X_test.flatten()]
     
#      #ns.fit(gram, mu)
#      #print(ns.best_score_, ns.best_params_, ns.best_estimator_)
     
#      crossval = cross_val_score(rs, gram[X.flatten()][:, X.flatten()], mu, cv=cv, scoring = f1)
#      ticfirst = time.perf_counter()
#      rs.fit(train_test, mu_train)
#      tocfirst = time.perf_counter()
#      #print(f"it took {tocfirst - ticfirst:0.4f} seconds")
#      predicted_test = rs.predict(test_test)
#      print("mathhews correlation coef: " ,  matthews_corrcoef (mu_test, predicted_test))
#      predict_train= rs.predict(train_test)
#      #print(f'fold {i}:')
#      #print('test')
#      print(classification_report(mu_test,  predicted_test))
#      #print('train')
#      #print(classification_report(mu_train,  predict_train))
#      #min_proba = rs.predict_proba(test_test)
#      print(crossval)
#      print("%0.2f F1 with a standard deviation of %0.2f" % (crossval.mean(), crossval.std()))
#      # conf = plot_confusion_matrix(rs, gram[X_test.flatten()][:, X_train.flatten()], mu_test,display_labels=['False','True'], colorbar = False)
#      # for labels in conf.text_.ravel():
#      #     labels.set_fontsize(30)
#      # plt.show()
# # full_view_min = np.concatenate([np.vstack((test_names,mu_test, predicted_test)).T,min_proba], axis = 1)
# # wrong_predictions_min = full_view_min[(full_view_min[:,1] != full_view_min[:,2])]
# # correct_predictions_min = full_view_min[(full_view_min[:,1] == full_view_min[:,2])]



# file_name=exp + '_100'

# X, gram, mu, names = get_rar_dataset(file_name+".csv")
# print('#############################################################################################################')


# print("set: ",file_name)

# for i in range(1):

#      X_train, X_test, mu_train, mu_test = train_test_split(X, mu, test_size=500-training_size, stratify=mu)

#      train_test = gram[X_train.flatten()][:, X_train.flatten()]
#      test_test = gram[X_test.flatten()][:, X_train.flatten()]
#      test_names = names[X_test.flatten()]
     
#      #ns.fit(gram, mu)
#      #print(ns.best_score_, ns.best_params_, ns.best_estimator_)
     
#      crossval = cross_val_score(rs, gram[X.flatten()][:, X.flatten()], mu, cv=cv, scoring = f1)
#      ticfirst = time.perf_counter()
#      rs.fit(train_test, mu_train)
#      tocfirst = time.perf_counter()
#      #print(f"it took {tocfirst - ticfirst:0.4f} seconds")
#      predicted_test = rs.predict(test_test)
#      print("mathhews correlation coef: " ,  matthews_corrcoef (mu_test, predicted_test))
#      predict_train= rs.predict(train_test)
#      #print(f'fold {i}:')
#      #print('test')
#      print(classification_report(mu_test,  predicted_test))
#      #print('train')
#      #print(classification_report(mu_train,  predict_train))
#      #min_proba = rs.predict_proba(test_test)
#      print(crossval)
#      print("%0.2f F1 with a standard deviation of %0.2f" % (crossval.mean(), crossval.std()))
#      # conf = plot_confusion_matrix(rs, gram[X_test.flatten()][:, X_train.flatten()], mu_test,display_labels=['False','True'], colorbar = False)
#      # for labels in conf.text_.ravel():
#      #     labels.set_fontsize(30)
#      # plt.show()
# # full_view_min = np.concatenate([np.vstack((test_names,mu_test, predicted_test)).T,min_proba], axis = 1)
# # wrong_predictions_min = full_view_min[(full_view_min[:,1] != full_view_min[:,2])]
# # correct_predictions_min = full_view_min[(full_view_min[:,1] == full_view_min[:,2])]





# file_name=exp + '_1000'

# X, gram, mu, names = get_rar_dataset(file_name+".csv")
# print('#############################################################################################################')

# print("set: ",file_name)

# for i in range(1):

#      X_train, X_test, mu_train, mu_test = train_test_split(X, mu, test_size=500-training_size, stratify=mu)

#      train_test = gram[X_train.flatten()][:, X_train.flatten()]
#      test_test = gram[X_test.flatten()][:, X_train.flatten()]
#      test_names = names[X_test.flatten()]
     
#      #ns.fit(gram, mu)
#      #print(ns.best_score_, ns.best_params_, ns.best_estimator_)
     
#      crossval = cross_val_score(rs, gram[X.flatten()][:, X.flatten()], mu, cv=cv, scoring = f1)
#      ticfirst = time.perf_counter()
#      rs.fit(train_test, mu_train)
#      tocfirst = time.perf_counter()
#      #print(f"it took {tocfirst - ticfirst:0.4f} seconds")
#      predicted_test = rs.predict(test_test)
#      print("mathhews correlation coef: " ,  matthews_corrcoef (mu_test, predicted_test))
#      predict_train= rs.predict(train_test)
#      #print(f'fold {i}:')
#      #print('test')
#      print(classification_report(mu_test,  predicted_test))
#      #print('train')
#      #print(classification_report(mu_train,  predict_train))
#      #min_proba = rs.predict_proba(test_test)
#      print(crossval)
#      print("%0.2f F1 with a standard deviation of %0.2f" % (crossval.mean(), crossval.std()))
#      # conf = plot_confusion_matrix(rs, gram[X_test.flatten()][:, X_train.flatten()], mu_test,display_labels=['False','True'], colorbar = False)
#      # for labels in conf.text_.ravel():
#      #     labels.set_fontsize(30)
#      # plt.show()
# # full_view_min = np.concatenate([np.vstack((test_names,mu_test, predicted_test)).T,min_proba], axis = 1)
# # wrong_predictions_min = full_view_min[(full_view_min[:,1] != full_view_min[:,2])]
# # correct_predictions_min = full_view_min[(full_view_min[:,1] == full_view_min[:,2])]



















    
