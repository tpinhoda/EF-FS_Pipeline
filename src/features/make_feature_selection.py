import logging
from os import environ
from os.path import join
from dotenv import find_dotenv, load_dotenv
from src.utils import utils
from src.features import baselines_feature_selection as baselines


def run(run_fs_baselines, ds_fold):
    logger_name = 'Feature Selection'
    logger = logging.getLogger(logger_name)
    # Find data.env automatically by walking up directories until it's found
    dotenv_path = find_dotenv(filename=join('parameters','feature_selection.env'))
     # Load up the entries as environment variables
    load_dotenv(dotenv_path)
    # Get dataset parameter
    input_filepath = environ.get('INPUT_DATASET')
    output_filepath = environ.get('OUTPUT_PATH')
    output_filepath = join(output_filepath,ds_fold)
    # Get data fold parameters
    type_folds = environ.get('TYPE_FOLDS')
    target_col = environ.get('TARGET')
    n_features = int(environ.get('FILTERING_N_FEATURES'))
    independent = environ.get('INDEPENDENT')
    output_filepath = utils.get_fold_type_folder_path(type_folds, output_filepath, logger_name)
    if type_folds == 'CN':
        n_neighbors = environ.get('C_N_NEIGHBORS')
        filter_train = environ.get('FILTER_TRAIN')
        center_candidate = environ.get('CENTER_CANDIDATE')
        folder_name = center_candidate + '_' + 'N'+n_neighbors+'_FT_'+filter_train
        output_filepath = join(output_filepath, folder_name)
        if filter_train == 'True':
            input_filepath = join(output_filepath, 'filtered_data.csv')

    output_filepath = utils.create_folder(output_filepath, 'experiments', logger_name)
    if n_features == -1:
        output_filepath = utils.create_folder(output_filepath, 'TG_{}_FN_CFS_IND_{}'.format(target_col, independent), logger_name)
    else:
        output_filepath = utils.create_folder(output_filepath, 'TG_{}_FN_{}_IND_{}'.format(target_col, str(n_features), independent),logger_name)
    
    # =============================================
    if run_fs_baselines == 'True' and independent == 'True':
        output_filepath = utils.create_folder(output_filepath, 'features_selected', logger_name)
        baselines.run(input_filepath, output_filepath, n_features, target_col)
    else:
        if independent == 'False':
            logger.warning('(Independent == False) Not running feature selection baselines.')
        else:
            logger.warning('Not running feature selection baselines.')