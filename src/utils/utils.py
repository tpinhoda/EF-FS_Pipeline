import logging
import pandas as pd
import lightgbm
import json
from os import mkdir
from os.path import join
from sklearn.linear_model import LinearRegression



def create_folder(path, folder_name, logger_name, show_msg=True):
    logger = logging.getLogger(logger_name)
    path = join(path, folder_name)
    if show_msg:
        logger.info('Creating folder: {}'.format(folder_name))
    try:
        mkdir(path)
    except FileExistsError:
        if show_msg:
            logger.info('Folder already exist: {}'.format(folder_name))
        else:
            pass
    return path


def get_fold_type_folder_path(type_folds, root_filepath, logger_name):
    logger = logging.getLogger(logger_name)
    if type_folds == 'R':
        root_filepath = join(root_filepath, 'Regiao')
    elif type_folds == 'S':
        root_filepath = join(root_filepath, 'UF')
    elif type_folds == 'ME':
        root_filepath = join(root_filepath, 'Meso')
    elif type_folds == 'MI':
        root_filepath = join(root_filepath, 'Micro')
    elif type_folds == 'D':
        root_filepath = join(root_filepath, 'Distrito')
    elif type_folds == 'SD':
        root_filepath = join(root_filepath, 'Subdistrito')
    elif type_folds == 'CN':
        root_filepath = join(root_filepath, 'Changing_Neighborhood')
    else:
        root_filepath = None
        logger.error('Incorrect type fold option try: [R, S, ME, MI, D, SD, CN]')
        exit()
    return root_filepath


def get_geo_attribute(type_folds, logger_name):
    logger = logging.getLogger(logger_name)
    if type_folds == 'R':
        geo_group = 'GEO_Cod_Grande_Regiao'
    elif type_folds == 'S':
        geo_group = 'GEO_Nome_UF'
    elif type_folds == 'ME':
        geo_group = 'GEO_Cod_Meso'
    elif type_folds == 'MI':
        geo_group = 'GEO_Cod_Micro'
    elif type_folds == 'D':
        geo_group = 'GEO_Cod_Distrito'
    elif type_folds == 'SD':
        geo_group = 'GEO_Cod_Subdistruto'
    else:
        geo_group = None
        logger.info('Incorrect type fold option try: [R, S, ME, MI, D, SD, CN]')
        exit()
    return geo_group


def get_model(model_name):
    if model_name == 'LR':
        model = LinearRegression()
    elif model_name == 'LGBM':
        model = lightgbm.LGBMRegressor()
    else:
        model = None
    return model

def get_features_from_file(path):
    with open(path) as f:
        selected_features = json.load(f)
    return selected_features['selected_features']

def split_data(data, target_col):
    x_data = data.drop(target_col, axis=1).copy()
    y_data = data[target_col].copy()
    return x_data, y_data


def make_data(fold_path, target_col, filename):
    train = pd.read_csv(join(fold_path, filename), low_memory=False)
    x_train, y_train = split_data(train, target_col)
    return x_train, y_train


def filter_by_selected_features(data, feature_selected):
    return data[feature_selected].copy()

def get_descriptive_attributes(data):
    census_cols = [c for c in data.columns if 'CENSUS' in c]
    idhm_cols = [c for c in data.columns if 'IDHM' in c]
    elections_cols = [c for c in data.columns if 'ELECTION' in c]
    
    elections_in_cols = [c for c in elections_cols if 'BOLSONARO' not in c]
    elections_in_cols = [c for c in elections_in_cols if 'HADDAD' not in c]
    elections_in_cols = [c for c in elections_in_cols if 'who_won' not in c]
    
    input_space = census_cols + idhm_cols + elections_in_cols
    return data[input_space]