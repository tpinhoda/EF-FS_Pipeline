import coloredlogs,  logging
coloredlogs.install()
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

import random
import pickle
from os import environ, listdir
from os.path import join
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm
from src.utils import utils

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


def train_model(folds_filepath, folds_names, model, features_selected, target_col, output_path):
    for fold in tqdm(folds_names, desc='Creating trained models by folds:'):
        x_train, y_train = utils.make_data(join(folds_filepath, fold), target_col, 'train.csv')
        x_train = utils.filter_by_selected_features(x_train, features_selected)
        model.fit(x_train, y_train)
        filename = fold + '.sav'
        save_model(model, output_path, filename)
        
           

def run(folds_filepath, exp_filepath, output_path, model_name, target_col):
    logger = logging.getLogger(__name__)
    
    folds_names = [fold for fold in listdir(folds_filepath)]
    fs_methods = [method for method in listdir(join(exp_filepath, 'features_selected'))]
    model = utils.get_model(model_name)
    for fs_method in fs_methods:
        logger.info('Creating {} models for: {}'.format(model_name, fs_method))
        selected_features = utils.get_features_from_file(join(exp_filepath, 'features_selected', fs_method))
        fs_method_output_path = utils.create_folder(output_path, fs_method.split('.')[0])
        train_model(folds_filepath, folds_names, model, selected_features, target_col, fs_method_output_path)
  
       
        
       