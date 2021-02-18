import coloredlogs,  logging
coloredlogs.install()
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

from os import environ
from os.path import join
from dotenv import find_dotenv, load_dotenv
from src.utils import utils
from src.model import make_prediction, make_train

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    # Find data.env automatically by walking up directories until it's found
    dotenv_path = find_dotenv(filename=join('parameters','spatial_cross_validation.env'))
    # Load up the entries as environment variables
    load_dotenv(dotenv_path)
    # Get path
    meshblocks_filepath = environ.get('MESHBLOCK')
    exp_filepath = environ.get('EXP_PATH')
    # Get data fold parameters
    type_folds = environ.get('TYPE_FOLDS')
    target_col = environ.get('TARGET')
    n_features = environ.get('FILTERING_N_FEATURES')
    random_perc = environ.get('RANDOM_PERC')
    # Get Spatial cross validation parameters
    scv_model_name = environ.get('MODEL_NAME')
    
    exp_filepath = utils.get_fold_type_folder_path(type_folds, exp_filepath)
    if type_folds == 'CN':
        n_neighbors = environ.get('C_N_NEIGHBORS')
        filter_train = environ.get('FILTER_TRAIN')
        center_candidate = environ.get('CENTER_CANDIDATE')
        folder_name = center_candidate + '_' + 'N'+n_neighbors+'_FT_'+filter_train
        exp_filepath = join(exp_filepath, folder_name)

    folds_filepath = join(exp_filepath, 'folds') 
    exp_filepath = utils.create_folder(exp_filepath, 'experiments')
    if n_features == '-1':
        exp_filepath = utils.create_folder(exp_filepath,'TG_{}_FN_{}_RP_{}'.format(target_col, 'CFS', random_perc))
    else:
        exp_filepath = utils.create_folder(exp_filepath,'TG_{}_FN_{}_RP_{}'.format(target_col, n_features, random_perc))
    
    output_path = utils.create_folder(exp_filepath, 'results')
    output_path = utils.create_folder(output_path, scv_model_name)
    models_path = utils.create_folder(output_path, 'models_trained')
    
    
    make_train.run(folds_filepath, exp_filepath, models_path, scv_model_name, target_col)
    results_by_folds = utils.create_folder(output_path, 'by_folds')
    make_prediction.run(folds_filepath, models_path, exp_filepath, results_by_folds, target_col)
    