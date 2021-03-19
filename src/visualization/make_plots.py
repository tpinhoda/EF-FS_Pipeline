from os import environ
from os.path import join
from dotenv import find_dotenv, load_dotenv
from fiona import env
from src.utils import utils
from src.visualization import make_map_plot, make_overall_plot, make_shap_plot

if __name__ == '__main__':
    # Find data.env automatically by walking up directories until it's found
    dotenv_path = find_dotenv(filename=join('parameters','make_plots.env'))
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
    # Get Dataset parameters
    index_col = environ.get('INDEX_COL')
    
    exp_filepath = utils.get_fold_type_folder_path(type_folds, exp_filepath)
    if type_folds == 'CN':
        n_neighbors = environ.get('C_N_NEIGHBORS')
        filter_train = environ.get('FILTER_TRAIN')
        center_candidate = environ.get('CENTER_CANDIDATE')
        folder_name = center_candidate + '_' + 'N'+n_neighbors+'_FT_'+filter_train
        exp_filepath = join(exp_filepath, folder_name)
        
    
    folds_filepath = join(exp_filepath, 'folds')
    exp_filepath = join(exp_filepath, 'experiments')
    if n_features == '-1':
        exp_filepath = join(exp_filepath,'TG_{}_FN_{}_RP_{}'.format(target_col, 'CFS', random_perc))
    else:
        exp_filepath = join(exp_filepath,'TG_{}_FN_{}_RP_{}'.format(target_col, n_features, random_perc))
    
    results_path = join(exp_filepath, 'results', scv_model_name)
    models_path = join(results_path, 'models_trained')
    results_by_folds = join(results_path, 'by_folds')
    plots_path = utils.create_folder(results_path, 'plots')
    
    
    make_overall_plot.run(results_by_folds, plots_path)
    map_plots_path = utils.create_folder(plots_path, 'map_plots')
    make_map_plot.run(folds_filepath, models_path, exp_filepath, map_plots_path, meshblocks_filepath, target_col, index_col, center_candidate)
    #shap_plots_path = utils.create_folder(plots_path, 'shap_plots')
    #make_shap_plot.run(folds_filepath, models_path, exp_filepath, shap_plots_path, target_col, index_col)