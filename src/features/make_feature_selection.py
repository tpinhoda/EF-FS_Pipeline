import logging
from os import environ, listdir
from os.path import join
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm
import weka.core.jvm as jvm
from src.utils import utils
from src.features import baselines_feature_selection as baselines


def run(run_fs_baselines, ds_fold):
    logger_name = 'Feature Selection'
    logger = logging.getLogger(logger_name)
    if run_fs_baselines == 'True':
        # Find data.env automatically by walking up directories until it's found
        dotenv_path = find_dotenv(filename=join('parameters','feature_selection.env'))
        # Load up the entries as environment variables
        load_dotenv(dotenv_path)
        # Get dataset parameter
        desc_filepath = environ.get('CENSUS_DESC')
        input_filepath = environ.get('INPUT_DATASET')
        output_filepath = environ.get('OUTPUT_PATH')
        output_filepath = join(output_filepath,ds_fold)
        # Get data fold parameters
        filter_data = environ.get('FILTER_DATA')
        type_folds = environ.get('TYPE_FOLDS')
        # Get feature selection parameter
        target_col = environ.get('TARGET')
        n_features = int(environ.get('FILTERING_N_FEATURES'))
        independent = environ.get('INDEPENDENT')
        output_filepath = utils.get_fold_type_folder_path(type_folds, output_filepath, logger_name)
        if filter_data == 'True':
            input_filepath = join(output_filepath, 'filtered_data.csv')
        if type_folds == 'CN':
            group_CN = environ.get('GROUP_CN')
            n_neighbors = environ.get('C_N_NEIGHBORS')
            filter_train = environ.get('FILTER_TRAIN')
            center_candidate = environ.get('CENTER_CANDIDATE')
            group_CN = environ.get('GROUP_CN')
            if group_CN != 'CN':
                geo_name = utils.get_name_geo_group(group_CN, logger_name)
                output_filepath = output_filepath +'_grouped_{}'.format(geo_name)
            
            folder_name = center_candidate + '_' + 'N'+n_neighbors+'_FT_'+filter_train
            output_filepath = join(output_filepath, folder_name)
            if filter_train == 'True':
                input_filepath = join(output_filepath, 'filtered_data.csv')
        folds_filepath = join(output_filepath, 'folds')
        output_filepath = utils.create_folder(output_filepath, 'experiments', logger_name)
        if n_features == -1:
            output_filepath = utils.create_folder(output_filepath, 'TG_{}_FN_CFS_IND_{}'.format(target_col, independent), logger_name)
        else:
            output_filepath = utils.create_folder(output_filepath, 'TG_{}_FN_{}_IND_{}'.format(target_col, str(n_features), independent),logger_name)
        
        output_desc_filepath = utils.create_folder(output_filepath, 'desc_features_selected', logger_name)
        output_filepath = utils.create_folder(output_filepath, 'features_selected', logger_name)
        
        # =============================================
        if independent == 'True':
            baselines.run(input_filepath, output_filepath, desc_filepath, output_desc_filepath, n_features, target_col)
        else:
            folds_names = [fold for fold in listdir(folds_filepath)]
            logger.info('Starting JVM.')
            logger.warning('Some warnings may appear.')
            jvm.start()
            for fold in tqdm(folds_names):
                output_foldpath = utils.create_folder(output_filepath, fold, logger_name)
                output_desc_foldpath = utils.create_folder(output_desc_filepath, fold, logger_name)
                input_filepath = join(folds_filepath, fold, 'train.csv')
                baselines.run(input_filepath, output_foldpath, desc_filepath, output_desc_foldpath, n_features, target_col, independent=False)
            jvm.stop()
    else:   
        logger.warning('Not running feature selection baselines.')