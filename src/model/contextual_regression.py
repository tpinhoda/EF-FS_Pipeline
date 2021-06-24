# imports
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats.stats import rankdata, kendalltau, spearmanr, weightedtau
from statistics import mean 
import math
import itertools
from sklearn.linear_model import LinearRegression
from src.utils import utils
from operator import itemgetter
from scipy.spatial import distance


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

    def __init__(self, model_name='LGBM'):
        self.model_name = model_name
        self.coord = dict()
        self.moldels_context = dict()
   
    def fit(self, x, y, context, geo_x, geo_y):
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
        self.context = context
       # count_context = context.value_counts()
       # for context_name, value in count_context.iteritems():
       #     self.w[context_name] = value * 100 /  len(context)
     
        
        x['context'] = context
        x['y'] = y
        x['GEO_x'] = geo_x
        x['GEO_y'] = geo_y
        self.data_group = x.groupby(by='context') 
        #for context, data in x.groupby(by='context'):
        #    self.coord[context] = {'x': data['GEO_x'].mean(), 'y': data['GEO_y'].mean() }
            #target = data['y']
            #train = data.drop(['y', 'context', 'GEO_x', 'GEO_y'], axis=1)
            #model = utils.get_model(self.model_name)
            #self.moldels_context[context] = model.fit(train, target)
        return self  
        
    def predict(self, x, geo_x, geo_y, fold_name):
        """ Predicts the value after the model has been trained.
        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Test samples
        Returns
        -------
        Predicted value
        """
        
        vizinhos = {
            'acre': 2,
            'amazonas': 5,
            'roraima': 2,
            'rondônia': 4,
            'pará': 6,
            'amapá': 1,
            'tocantins':6,
            'mato grosso':6,
            'goiás':7,
            'mato grosso do sul':5,
            'distrito federal':1,
            'paraná':3,
            'santa catarina':2,
            'rio grande do sul':1,
            'são paulo':5,
            'rio de janeiro':3,
            'espã\xadrito santo':3,
            'minas gerais':5,
            'bahia':9,
            'sergipe':2,
            'alagoas':3,
            'pernambuco':5,
            'paraiba':5,
            'rio grande do norte':5,
            'ceará':4,
            'piauí':4,
            'maranhão':4
        }
        
        kl_dist = dict()
        geo_dist = dict()
        x['GEO_x'] = (geo_x/100) * 2
        x['GEO_y'] = (geo_y/100) * 2
        for context in self.context:
            #coord = self.coord[context]
            #geo_dist[context] = (coord['x'] - geo_x)**2 + (coord['y'] - geo_y)**2
            x_context = self.data_group.get_group(context).drop(['y', 'context'], axis=1)
            x_context['GEO_x'] = (x_context['GEO_x'] / 100) * 2
            x_context['GEO_y'] = (x_context['GEO_y'] / 100) * 2
            kl_dist[context] = distance.minkowski(x.mean(), x_context.mean(), 2)
            #kl_dist[context] = KLdivergence(x.mean(axis=0), x_context.mean(axis=0))
            
            #chose_context = min(geo_dist.keys(), key=(lambda k: geo_dist[k]))
            #y_pred = self.moldels_context[chose_context].predict(x)
            # return y_pred
        x.drop(['GEO_x', 'GEO_y'], axis=1, inplace=True)
        x_context.drop(['GEO_x', 'GEO_y'], axis=1, inplace=True)
         #chose_context = sorted(geo_dist.items(), key=itemgetter(1))
        kl_choose = sorted(kl_dist.items(), key=itemgetter(1))
            #print('context: {}'.format(chose_context))
            #print('Context: {}'.format([c for c in chose_context]))
        results = pd.DataFrame()
        list_groups = []
        minority_perc = 0.15
        percs_won = 0
        percs_lost = 0
        len_train = 2.4
        list_context = []
        for context, _ in kl_choose:
            if  (percs_won <= minority_perc or percs_lost <= minority_perc) or vizinhos[fold_name] * len_train > len(list_context): 
                list_context.append(context)
                list_groups.append(self.data_group.get_group(context))
                train = pd.concat(list_groups, axis=0)
                #len_train = len(train)
                train['won_lost'] = np.where(train['y'] > 50, 'won', 'lost')
                percs = train['won_lost'].value_counts(normalize=True)
                try:
                    percs_won = percs['won']
                except KeyError:
                    percs_won = 0
                try:
                    percs_lost = percs['lost']
                except KeyError:
                    percs_lost = 0 
                
                #results[context] = self.moldels_context[context].predict(x)
                #results[context] = results[context]
        #print(list_context)
        #print(train['won_lost'].value_counts(normalize=True))
        train.drop(['GEO_x', 'GEO_y', 'context', 'won_lost'], axis=1, inplace=True)
        #print('Len: {}'.format(len(train)))
        y = train['y']
        model = utils.get_model(self.model_name)
        model.fit(train.drop('y', axis=1), y)
        y_pred = model.predict(x)
        return y_pred
            

    
    
    
def KLdivergence(x, y):
  """Compute the Kullback-Leibler divergence between two multivariate samples.
  Parameters
  ----------
  x : 2D array (n,d)
    Samples from distribution P, which typically represents the true
    distribution.
  y : 2D array (m,d)
    Samples from distribution Q, which typically represents the approximate
    distribution.
  Returns
  -------
  out : float
    The estimated Kullback-Leibler divergence D(P||Q).
  References
  ----------
  Pérez-Cruz, F. Kullback-Leibler divergence estimation of
continuous distributions IEEE International Symposium on Information
Theory, 2008.
  """
  from scipy.spatial import cKDTree as KDTree

  # Check the dimensions are consistent
  x = np.atleast_2d(x)
  y = np.atleast_2d(y)

  n,d = x.shape
  m,dy = y.shape

  assert(d == dy)


  # Build a KD tree representation of the samples and find the nearest neighbour
  # of each point in x.
  xtree = KDTree(x)
  ytree = KDTree(y)

  # Get the first two nearest neighbours for x, since the closest one is the
  # sample itself.
  r = xtree.query(x, k=2, eps=.01, p=2)[0][:,1]
  s = ytree.query(x, k=1, eps=.01, p=2)[0]

  # There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
  # on the first term of the right hand side.
  return -np.log(r/s).sum() * d / n + np.log(m / (n - 1.))