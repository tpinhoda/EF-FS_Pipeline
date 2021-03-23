
import logging
import pandas as pd
import pickle
from os import listdir
from os.path import join
from tqdm import tqdm
from src.utils import utils
from sklearn.metrics import mean_squared_error
from scipy.stats import weightedtau, kendalltau, spearmanr, rankdata

def create_eval_dict():
    eval_dict = {'fold_name': [],
                 'size': [],
                 'n_features': [],
                 'sae': [],
                 'rmse': [],
                 'kendall': [],
                 'wkendall': [],
                 'spearmanr': [],
                 'hit_center': [],
                 'rank_dist_center': [],
                 }
    return eval_dict


def calculate_sae(model, x, y_true):
    y_pred = model.predict(x)
    OldRange = (y_pred.max() - y_pred.min())
    NewRange = (y_true.max() - y_true.min())
    y_pred = (((y_pred - y_pred.min()) * NewRange)/OldRange) + y_true.min()
    sae =  abs(y_pred - y_true).sum()
    return sae

def calculate_rmse(model, x, y_true):
    y_pred = model.predict(x)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return rmse

def calculate_wkendall(model, x, y_true):
    y_pred = model.predict(x)
    tau, _ = weightedtau(y_true, y_pred)
    return tau


def calculate_kendall(model, x, y_true):
    y_pred = model.predict(x)
    tau, _ = kendalltau(y_true, y_pred)
    return tau


def calculate_spearman(model, x, y_true):
    y_pred = model.predict(x)
    ro, _ = spearmanr(y_true, y_pred)
    return ro


def check_center_neighbor(model, x, y_true, center_neighbor):
    y_pred = model.predict(x)
    rank_pred = pd.Series(rankdata(y_pred, method='ordinal'), index=x.index, name='rank_pred')
    rank_true = pd.Series(rankdata(y_true, method='ordinal'), index=x.index, name='rank_true')
    center = [local_idx for local_idx in center_neighbor.index if center_neighbor[local_idx] == 'center']
    if rank_pred[center[0]] == rank_true[center[0]]:
        return 1
    else:
        return 0

def calculate_rank_dist_center(model, x, y_true, center_neighbor):
    y_pred = model.predict(x)
    rank_pred = pd.Series(rankdata(y_pred, method='ordinal'), index=x.index, name='rank_pred')
    rank_true = pd.Series(rankdata(y_true, method='ordinal'), index=x.index, name='rank_true')
    center = [local_idx for local_idx in center_neighbor.index if center_neighbor[local_idx] == 'center']
    return abs(rank_true[center[0]] - rank_pred[center[0]])
    
    

def model_predict(folds_filepath, folds_names, models_path,  output_path, fs_method, selected_features, target_col):
    metrics = create_eval_dict()
    for fold_name in tqdm(folds_names, desc='Predicting folds:', position=0, leave=False):
        
        x_test, y_test = utils.make_data(join(folds_filepath, fold_name), target_col, 'test.csv')
        if 'Changing_Neighborhood' in output_path:
            center_neighbor = x_test['center_neighbor']
        
        x_test = utils.filter_by_selected_features(x_test, selected_features)
        model = load_model(join(models_path, fs_method, fold_name + '.sav'))

        
        metrics['fold_name'].append(fold_name)
        metrics['size'].append(len(y_test))
        metrics['n_features'].append(len(selected_features))
        metrics['sae'].append(calculate_sae(model, x_test, y_test))
        metrics['rmse'].append(calculate_rmse(model, x_test, y_test))
        metrics['kendall'].append(calculate_kendall(model, x_test, y_test))
        metrics['wkendall'].append(calculate_wkendall(model, x_test, y_test))
        metrics['spearmanr'].append(calculate_spearman(model, x_test, y_test))
        if 'Changing_Neighborhood' in output_path:
            metrics['hit_center'].append(check_center_neighbor(model, x_test, y_test, center_neighbor))
            metrics['rank_dist_center'].append(calculate_rank_dist_center(model, x_test, y_test, center_neighbor))
        else:
            metrics['hit_center'].append(0)
            metrics['rank_dist_center'].append(0)
            
        
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(join(output_path, fs_method +'.csv'), index=False)
    

def load_model(filepath):
    return pickle.load(open(filepath, 'rb'))
   
def run(folds_filepath, models_path, exp_filepath,  output_path, target_col):
    logger_name = 'Evaluation'
    logger = logging.getLogger(logger_name)
    fs_methods = [fs_method for fs_method in listdir(models_path)]
    folds_names = [fold_name for fold_name in listdir(folds_filepath)]
    for fs_method in fs_methods:
        logger.info('Predicting using method: {}'.format(fs_method))
        selected_features = utils.get_features_from_file(join(exp_filepath, 'features_selected', fs_method + '.json'))
        model_predict(folds_filepath, folds_names, models_path, output_path, fs_method, selected_features, target_col) 
          
            
        