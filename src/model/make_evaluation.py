import logging
from os import environ
from os.path import join
from dotenv import find_dotenv, load_dotenv
from fiona import env
from src.utils import utils
from src.model import make_prediction, make_train

def run(run_train, run_prediction, ds_folds):
    logger_name = 'Evaluation'
    logger = logging.getLogger(logger_name)
    # Get path
    exp_filepath = environ.get('OUTPUT_PATH')
    exp_filepath = join(exp_filepath, ds_folds)
    # Get data fold parameters
    type_folds = environ.get('TYPE_FOLDS')
    target_col = environ.get('TARGET')
    n_features = environ.get('FILTERING_N_FEATURES')
    independent = environ.get('INDEPENDENT')
    # Get Spatial cross validation parameters
    scv_model_name = environ.get('MODEL_NAME')
    
    exp_filepath = utils.get_fold_type_folder_path(type_folds, exp_filepath, logger_name)
    if type_folds == 'CN':
        n_neighbors = environ.get('C_N_NEIGHBORS')
        filter_train = environ.get('FILTER_TRAIN')
        center_candidate = environ.get('CENTER_CANDIDATE')
        folder_name = center_candidate + '_' + 'N'+n_neighbors+'_FT_'+filter_train
        group_CN = environ.get('GROUP_CN')
        if group_CN != 'CN':
            geo_name = utils.get_name_geo_group(group_CN, logger_name)
            exp_filepath= exp_filepath +'_grouped_{}'.format(geo_name)
        exp_filepath = join(exp_filepath, folder_name)

    folds_filepath = join(exp_filepath, 'folds')
    if run_train == 'True':
        exp_filepath = utils.create_folder(exp_filepath, 'experiments', logger_name)
        if n_features == '-1':
            exp_filepath = utils.create_folder(exp_filepath,'TG_{}_FN_{}_IND_{}'.format(target_col, 'CFS', independent), logger_name)
        else:
            exp_filepath = utils.create_folder(exp_filepath,'TG_{}_FN_{}_IND_{}'.format(target_col, n_features, independent), logger_name)
        
        output_path = utils.create_folder(exp_filepath, 'results', logger_name)
        output_path = utils.create_folder(output_path, scv_model_name, logger_name)
        models_path = utils.create_folder(output_path, 'models_trained', logger_name)
        make_train.run(folds_filepath, exp_filepath, models_path, scv_model_name, target_col, independent)
    else:
        logger.warning('Not training the model.')
    if run_prediction == 'True':
        if run_train == 'False':
            exp_filepath = join(exp_filepath, 'experiments')
            if n_features == '-1':
                exp_filepath = join(exp_filepath,'TG_{}_FN_{}_IND_{}'.format(target_col, 'CFS', independent))
            else:
                exp_filepath = join(exp_filepath,'TG_{}_FN_{}_IND_{}'.format(target_col, n_features, independent)) 
            output_path = join(exp_filepath, 'results', scv_model_name)   
            models_path = join(output_path, 'models_trained')
        
        results_by_folds = utils.create_folder(output_path, 'by_folds', logger_name)
        make_prediction.run(scv_model_name, folds_filepath, models_path, exp_filepath, results_by_folds, target_col, independent)
    else:
        logger.warning('Not running prediction.')
        
    