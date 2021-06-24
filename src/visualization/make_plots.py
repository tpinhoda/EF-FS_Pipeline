import logging
from os import environ
from os.path import join
from dotenv import find_dotenv, load_dotenv
from src.utils import utils
from src.visualization import make_map_plot, make_overall_plot, make_shap_plot

def run(run_general_maps, run_map_plots, run_shap_plots, ds_fold):
    logger_name = 'Visualization'
    logger = logging.getLogger(logger_name)
    # Find data.env automatically by walking up directories until it's found
    dotenv_path = find_dotenv(filename=join('parameters','make_plots.env'))
    # Load up the entries as environment variables
    load_dotenv(dotenv_path)
    # Get path
    meshblocks_filepath = environ.get('MESHBLOCK')
    exp_filepath = environ.get('OUTPUT_PATH')
    exp_filepath = join(exp_filepath, ds_fold)
    # Get data fold parameters
    type_folds = environ.get('TYPE_FOLDS')
    geo_groups = environ.get('GEO_GROUP') 
    target_col = environ.get('TARGET')
    n_features = environ.get('FILTERING_N_FEATURES')
    independent = environ.get('INDEPENDENT')
    center_candidate = environ.get('CENTER_CANDIDATE')
    # Get Spatial cross validation parameters
    scv_model_name = environ.get('MODEL_NAME')
    # Get Dataset parameters
    index_col = environ.get('INDEX_COL')
    
    exp_filepath = utils.get_type_fold_path(type_folds, exp_filepath, logger_name)
    if type_folds == 'CN':
        n_neighbors = environ.get('C_N_NEIGHBORS')
        filter_train = environ.get('FILTER_TRAIN')
        folder_name = center_candidate + '_' + 'N'+n_neighbors+'_FT_'+filter_train
        group_CN = environ.get('GROUP_CN')
        if group_CN != 'CN':
            geo_name = utils.get_name_geo_group(group_CN, logger_name)
            exp_filepath = exp_filepath +'_grouped_{}'.format(geo_name)
        exp_filepath = join(exp_filepath, folder_name)
    else:
        exp_filepath = utils.get_geo_group_folder_path(geo_groups, exp_filepath, logger_name)
        
        
    
    folds_filepath = join(exp_filepath, 'folds')
    exp_filepath = join(exp_filepath, 'experiments')
    if n_features == '-1':
        exp_filepath = join(exp_filepath,'TG_{}_FN_{}_IND_{}'.format(target_col, 'CFS', independent))
    else:
        exp_filepath = join(exp_filepath,'TG_{}_FN_{}_IND_{}'.format(target_col, n_features, independent))
    
    results_path = join(exp_filepath, 'results', scv_model_name)
    models_path = join(results_path, 'models_trained')
    results_by_folds = join(results_path, 'by_folds')
    
    if run_general_maps == 'True' or run_map_plots == 'True' or run_shap_plots == 'True':
        plots_path = utils.create_folder(results_path, 'plots', logger_name)
    
    if run_general_maps == 'True':
        make_overall_plot.run(results_by_folds, plots_path)
    else:
        logger.warning('Not generating general plots.')
    if run_map_plots == 'True':
        map_plots_path = utils.create_folder(plots_path, 'map_plots', logger_name)
        make_map_plot.run(folds_filepath, models_path, exp_filepath, map_plots_path, meshblocks_filepath, target_col, index_col, center_candidate, independent, scv_model_name)
    else:
        logger.warning('Not generating map plots.')
    if run_shap_plots == 'True':
        shap_plots_path = utils.create_folder(plots_path, 'shap_plots', logger_name)
        make_shap_plot.run(folds_filepath, models_path, exp_filepath, shap_plots_path, target_col, index_col)
    else:
        logger.warning('Not generating shap plots.')