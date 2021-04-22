import logging
import random
import pickle
import numpy as np
from os import environ, listdir, chdir, getcwd
from os.path import join
from tqdm import tqdm
from src.utils import utils

from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW

def get_rep_folds(list_folds, k, rep):
    repetition_cv = dict()
    for i in range(rep):
        if k == 'all':
            repetition_cv[str(i)] = list_folds
        else:
            repetition_cv[str(i)] = random.sample(list_folds, int(k))
        
        list_folds = [f for f in list_folds if f not in repetition_cv[str(i)]]
    return repetition_cv


def save_model(model, path, filename):
     pickle.dump(model, open(join(path, filename), 'wb'))
        
def train_model(model_name, folds_filepath, fold, model, features_selected, target_col, output_path, meshblock=[], index_col=None):
    x_train, y_train = utils.make_data(join(folds_filepath, fold), target_col, 'train.csv')
    if model_name == 'CR':
        geo_x = x_train['GEO_x']
        geo_y = x_train['GEO_y']
        context = x_train['GEO_Nome_UF']
        x_train = utils.filter_by_selected_features(x_train, features_selected)
        model.fit(x_train, y_train, context, geo_x, geo_y)
    elif model == 'GWR':
        coords = utils.get_geocoordinates(x_train)
        x_train = utils.filter_by_selected_features(x_train, features_selected)
        model = GWR(coords, y_train.values.reshape((-1,1)), x_train.values, bw=40.000, fixed=False, kernel='gaussian')
    else:
        x_train = utils.filter_by_selected_features(x_train, features_selected)
        model.fit(x_train, y_train)

        #gwr_selector = Sel_BW(coords, y_train.values.reshape((-1,1)), x_train.values)
        #gwr_bw = gwr_selector.search(bw_min=2)
        #print(gwr_bw)
    filename = fold + '.sav'
    save_model(model, output_path, filename)
        
           

def run(folds_filepath, exp_filepath, output_path, model_name, target_col, independent):
    logger_name = 'Evaluation'
    logger = logging.getLogger(logger_name)
    
    folds_names = [fold for fold in listdir(folds_filepath)]
    
    if model_name == 'GWR':
        model = None
    else:
        model = utils.get_model(model_name)
            
    if independent == 'True':
        fs_methods = [method for method in listdir(join(exp_filepath, 'features_selected'))]
        for fs_method in fs_methods:
            logger.info('Creating {} models for: {}'.format(model_name, fs_method))
            selected_features = utils.get_features_from_file(join(exp_filepath, 'features_selected', fs_method))
            fs_method_output_path = utils.create_folder(output_path, fs_method.split('.')[0], logger_name)
            for fold in tqdm(folds_names, desc='Creating trained models by folds:', leave=False):
                train_model(model_name, folds_filepath, fold, model, selected_features, target_col, fs_method_output_path)
            exit()
    else:
        for fold in folds_names:
            fs_methods = [method for method in listdir(join(exp_filepath, 'features_selected', fold))]
            logger.info('Creating {} models for: {}'.format(model_name, fold))
            for fs_method in tqdm(fs_methods, desc='Training:', leave=False):
                selected_features = utils.get_features_from_file(join(exp_filepath, 'features_selected', fold, fs_method))
                fs_method_output_path = utils.create_folder(output_path, fs_method.split('.')[0], logger_name, show_msg=False)
                train_model(model_name, folds_filepath, fold, model, selected_features, target_col, fs_method_output_path)
            
        
        
  
       
        
       