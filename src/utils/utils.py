import logging
import pandas as pd
import lightgbm
import json
from os import mkdir
from os.path import join
from sklearn.linear_model import LinearRegression



def create_folder(path, folder_name):
    logger = logging.getLogger(__name__)
    path = join(path, folder_name)
    logger.info('Creating folder: {}'.format(folder_name))
    try:
        mkdir(path)
    except FileExistsError:
        logger.info('Folder already exist: {}'.format(folder_name))
    return path


def get_fold_type_folder_path(type_folds, root_filepath):
    logger = logging.getLogger(__name__)
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
        logger.info('Incorrect type fold option try: [R, S, ME, MI, D, SD, CN]')
        exit()
    return root_filepath


def get_geo_attribute(type_folds):
    logger = logging.getLogger(__name__)
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