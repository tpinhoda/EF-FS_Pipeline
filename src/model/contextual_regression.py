# imports
import numpy as np
import pandas as pd
from scipy.stats.stats import rankdata, kendalltau, spearmanr, weightedtau
from statistics import mean 
import math
import itertools
from sklearn.linear_model import LinearRegression
from src.utils import utils
from operator import itemgetter


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
        for context, data in x.groupby(by='context'):
            self.coord[context] = {'x': data['GEO_x'].mean(), 'y': data['GEO_y'].mean() }
            target = data['y']
            train = data.drop(['y', 'context', 'GEO_x', 'GEO_y'], axis=1)
            model = utils.get_model(self.model_name)
            self.moldels_context[context] = model.fit(train, target)
        return self  
        
    def predict(self, x, geo_x, geo_y):
        """ Predicts the value after the model has been trained.
        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Test samples
        Returns
        -------
        Predicted value
        """
        kl_dist = dict()
        geo_dist = dict()
        if len(x) > 1:
            for context in self.context:
                coord = self.coord[context]
                geo_dist[context] = (coord['x'] - geo_x)**2 + (coord['y'] - geo_y)**2
                x_context = self.data_group.get_group(context).drop(['y', 'context', 'GEO_x', 'GEO_y'], axis=1)
                kl_dist[context] = KLdivergence(x, x_context)
            
            #chose_context = min(geo_dist.keys(), key=(lambda k: geo_dist[k]))
            #y_pred = self.moldels_context[chose_context].predict(x)
        # return y_pred
        
            chose_context = sorted(geo_dist.items(), key=itemgetter(1))[:5]
            kl_choose = sorted(kl_dist.items(), key=itemgetter(1))[:5] 
            print('context: {}'.format(chose_context))
            print('KL: {}'.format(kl_choose))
            results = pd.DataFrame()
            for context, _ in kl_choose:
                results[context] = self.moldels_context[context].predict(x)
                results[context] = results[context]
            
            results['pred'] = results.mean(axis=1) 
            #results['pred'] = (results.sum(axis=1 )/100)
            return results['pred'].values
    
    
    
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