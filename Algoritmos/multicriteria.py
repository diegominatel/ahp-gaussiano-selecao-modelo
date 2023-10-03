import numpy as np
import pandas as pd
import math

from itertools import combinations

''' Escala de Radom Index (RI) por número de variáveis (retirada de Wharton (1980) '''
RI = {
    3 :  0.58,
    4 :  0.90,
    5 :  1.12,
    6 :  1.24,
    7 :  1.32,
    8 :  1.41,
    9 :  1.45, 
    10 : 1.49, 
    11 : 1.51
}

class Multicriteria:
    def __init__(self, criteria_list, inverted_criteria = []):
        self.criteria = criteria_list
        self.inverted_criteria = inverted_criteria
        self.results = None
    
    def ranking(self):
        return self.results.sort_values(by='value', ascending=False).reset_index(drop=True)
    
class Traditional(Multicriteria):
    def __init__(self, criteria_list, inverted_criteria = []):
        super().__init__(criteria_list, inverted_criteria)
        
    def calculate(self, aux_decision_matrix):
        decision_matrix = aux_decision_matrix[self.criteria].copy()
        ''' Cria o Frame com os resultados '''
        self.results = pd.DataFrame(columns=['item', 'value'])
        ''' Se existe alguma coluna que a importância é do maior para o maior, ajusta os valores '''
        ''' [0, 1] '''
        decision_matrix[self.inverted_criteria] = 1 - decision_matrix[self.inverted_criteria]
        ''' Calcula os resultados '''
        self.results['item'] = decision_matrix.index
        self.results['value'] = list(decision_matrix.sum(axis=1))
        return self.results['value']

class MCPM(Multicriteria):
    def __init__(self, criteria_list, inverted_criteria = []):
        super().__init__(criteria_list, inverted_criteria)
        
    def multicriteria(self, values):
        pairs = list(combinations(values, 2))
        area = 0
        for a, b in pairs:
            area += (a*b*math.sin((2*math.pi)/3)/2)    
        return area

    def calculate(self, aux_decision_matrix):
        decision_matrix = aux_decision_matrix[self.criteria].copy()
        ''' Cria o Frame com os resultados '''
        self.results = pd.DataFrame(columns=['item', 'value'])
        ''' Se existe alguma coluna que a importância é do maior para o maior, ajusta os valores '''
        ''' [0, 1] '''
        decision_matrix[self.inverted_criteria] = 1 - decision_matrix[self.inverted_criteria]
        ''' Adapta para o formato do cálculo '''
        matrix = decision_matrix[self.criteria].to_numpy()
        ''' Copia os resultados no frame de resultados '''
        self.results['item'] = decision_matrix.index
        self.results['value'] = [self.multicriteria(row) for row in matrix]
        return self.results['value']
                
class GaussianAHP(Multicriteria):
    def __init__(self, criteria_list, inverted_criteria = []):
        super().__init__(criteria_list, inverted_criteria)
        
    def calculate(self, aux_decision_matrix):
        decision_matrix = aux_decision_matrix[self.criteria].copy()
        ''' Cria o Frame com os resultados '''
        self.results = pd.DataFrame(columns=['item', 'value'])
        #''' Se existe alguma coluna que a importância é do maior para o maior, ajusta os valores '''
        #decision_matrix[self.inverted_criteria] = 1/decision_matrix[self.inverted_criteria]
        ''' Se existe alguma coluna que a importância é do maior para o maior, ajusta os valores '''
        ''' [0, 1] '''
        decision_matrix[self.inverted_criteria] = 1 - decision_matrix[self.inverted_criteria]
        ''' Normaliza valores '''
        decision_matrix = decision_matrix/decision_matrix.sum()
        ''' Fator Gaussiano Normalizado '''
        ngf = (decision_matrix.std()/decision_matrix.mean())/(decision_matrix.std()/decision_matrix.mean()).sum()
        ''' Realiza o cálculo '''
        decision_matrix = decision_matrix*ngf
        ''' Copia os resultados no frame de resultados '''
        self.results['item'] = decision_matrix.index
        self.results['value'] = list(decision_matrix.sum(axis=1))
        return self.results['value']
        
class AHP(Multicriteria):
    
    def __init__(self, criteria_list, criteria_weights, inverted_criteria = [], cr_threshold=0.10):
        super().__init__(criteria_list, inverted_criteria)
        self.criteria_weights = criteria_weights
        self.cr_threshold = cr_threshold
        self.n = len(self.criteria)
        self.cr = None
        self.priority_vector = None
        self.consistency_check()
        
    def get_cr(self):
        return self.cr
    
    def get_priority_vector(self):
        return self.priority_vector
    
    def consistency_check(self):
        ''' Realiza a checagem de consitência na matriz de pesos '''
        self.criteria_normalized = self.criteria_weights.copy()
        ''' Normaliza a matriz'''
        for column in self.criteria_normalized.columns:
            self.criteria_normalized[column] = self.criteria_normalized[column]/self.criteria_normalized[column].sum()
        ''' Calcula o vetor de prioridade '''
        self.priority_vector = []
        for row in self.criteria_normalized.index:
            self.priority_vector.append(self.criteria_normalized.loc[row].sum()/self.n)
        ''' Calcula os critérios para os cálculos do lambda'''
        self.criteria_analysis = self.criteria_weights.copy()
        for i, column in enumerate(self.criteria_analysis.columns):
            self.criteria_analysis[column] = self.criteria_analysis[column]*self.priority_vector[i]
        ''' Calcula o lambda e o cr '''
        lambdas = []
        for i, row in enumerate(self.criteria_analysis):
            lambdas.append(self.criteria_analysis.loc[row].sum()/self.priority_vector[i])
        max_lambda = sum(lambdas)/len(lambdas)
        ci = (max_lambda - self.n)/(self.n - 1)
        ri = RI[self.n]
        self.cr = ci/ri
        ''' Verifica a razão de consitência '''
        if self.cr > self.cr_threshold:
            raise ValueError('A razão de consistência (%.2f%%) é maior do que %.2f%%' % (self.cr*100, 
                                                                                         self.cr_threshold*100))                 
    
    def calculate(self, aux_decision_matrix):
        decision_matrix = aux_decision_matrix.copy()
        ''' Cria o Frame com os resultados'''
        self.results = pd.DataFrame(columns=['item', 'probability'])
        ''' Se existe alguma coluna que a importância é do maior para o maior, ajusta os valores '''
        decision_matrix[self.inverted_criteria] = 1/decision_matrix[self.inverted_criteria]
        ''' Realiza o cálculo'''
        decision_matrix = (decision_matrix/decision_matrix.sum())*self.priority_vector
        ''' Copia os resultados no frame de resultados '''
        self.results['item'] = decision_matrix.index
        self.results['probability'] = list(decision_matrix.sum(axis=1))
        