# imports
import numpy as np
import pandas as pd
from scipy.stats.stats import rankdata, kendalltau, spearmanr, weightedtau
from statistics import mean 
import math
import itertools
from sklearn.linear_model import LinearRegression
from src import utils


class ContextualRegression:
    """Linear Regression Using Gradient Descent.
    Parameters
    ----------
    Model : object
        regression model
    Attributes
    ----------
    models_contexts
    """

    def __init__(self, model_name='LGBM')):
        self.model_name = model_name
   
    def fit(self, x, y, context):
        """Fit the training data
        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Training samples
        y : array-like, shape = [n_samples, n_target_values]
            Target values
        context: array-like, shape = [n_samples]
            Context values
        Returns
        -------
        self : object
        """
        self.moldels_context = dict()
        x['context'] = context
        x['y'] = y
        for context, data in x.groupby(by='context'):
            target = data['y']
            train = data.drop(['y', 'context'], axis=1)
            model = utils.get_model(self.model_name)
            self.moldels_context[context] = model.fit(train, target)
        return self  
        
    def predict(self, x):
        """ Predicts the value after the model has been trained.
        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Test samples
        Returns
        -------
        Predicted value
        """
        results = pd.DataFrame()
        for context in self.moldels_context:
            results[context] = self.moldels_context[context].predict(x)
        
        print(results)
        print(results.mean(axis=1 ))
        exit()     
        return 

    
def loss_function(y_pred, y_true):
    # sorted_rank = rankdata(-y_pred, method='ordinal')
    # sorted_rank = pd.DataFrame(sorted_rank)
    # rank_votes = rankdata(-y_true, method='ordinal')
    # rank_votes = pd.DataFrame(rank_votes)
    # sub = sorted_rank != rank_votes
    # tau = sub[0].sum()
    # print(tau)
    tau, _ = kendalltau(pd.DataFrame(y_pred), pd.DataFrame(y_true))
    #A = y_true.copy().values.tolist()
    #B = y_pred.copy().tolist()
    #tau = kendallTau(A, B)
   # tau = scaled_absolute_error(y_pred, y_true)
    if math.isnan(tau):
        tau = 0
    
    return tau


def objective_function_local(beta, x, y):
    all_errors = []
    for label, x_data in x.groupby('labels'):
        if label != -1:
            y_data = y[y['labels'] == label].copy()
            y_data.drop('labels', axis=1, inplace=True)
            x_data.drop('labels', axis=1, inplace=True)
            error = loss_function(np.dot(x_data, beta), y_data)
            #print('{}: {}'.format(label, error))
            all_errors.append(-error)
        
    #print('ALL: {}'.format(mean(all_errors)))
    #print(beta)
    
    return mean(all_errors)


def objective_function(beta, x, y):
    error = loss_function(np.dot(x, beta), y)
    #print(error)
    return -error


def kendallTau(A, B):
    A = rankdata([-v[0] for v in A])
    B = rankdata([-v for v in B])
    pairs = itertools.combinations(range(0, len(A)), 2)

    distance = 0
    for x, y in pairs:
        a = A[x] - A[y]
        b = B[x] - B[y]

        # if discordant (different signs)
        if (a * b < 0):
            distance += 1
    return distance


def scaled_absolute_error(y_pred, y_true):
    y_pred = pd.DataFrame(y_pred)
    OldRange = (y_pred[y_pred.columns[0]].max() - y_pred[y_pred.columns[0]].min())
    NewRange = (y_true[y_true.columns[0]].max() - y_true[y_true.columns[0]].min())
    y_pred = (((y_pred[y_pred.columns[0]] - y_pred[y_pred.columns[0]].min()) * NewRange)/OldRange) + y_true[y_true.columns[0]].min()
    sae = abs(y_pred - y_true[y_true.columns[0]]).sum()
    return sae
