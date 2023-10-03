import math
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def set_configs(n_columns):
    
    decision_tree = {
        'DT' : [DecisionTreeClassifier,
                {'criterion' : ['gini'],
                 'min_samples_leaf' : list(range(1, 31)),
                 'min_samples_split' : [5], 
                 'random_state' : [42]}]
    }
    
    knn = {
        'KNN' : [KNeighborsClassifier,
                 {'n_neighbors' : list(range(3, 62, 2))}]
    }
    
    mlp = {
        'MLP' : [MLPClassifier,
                 {'hidden_layer_sizes' : list(range(5, 35, 1)),
                  'random_state' : [42]}]
    }
    
    random_forest = {
        'RF' : [RandomForestClassifier,
                {'n_estimators' : list(range(50, 500, 15)),
                 'min_samples_split' : [math.floor(abs(math.sqrt(n_columns - 1)))], 
                 'random_state' : [42]}]
    }
    
    svm = {
        'SVM' : [SVC,
                 {'kernel' : ['rbf'], 'C' : [1], 'gamma' : list(np.arange(0.0025, 0.75, 0.025)), 
                  'random_state' : [42]}]
    }
    
    
    all_configs = {
        'dt'    : decision_tree,
        'knn'   : knn,
        'mlp'   : mlp,
        'rf'    : random_forest,
        'svm'   : svm
    }
    
    return all_configs