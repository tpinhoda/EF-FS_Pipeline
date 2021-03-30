import coloredlogs,  logging
coloredlogs.install()
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

from pathlib import Path
from os import environ
from os.path import join
from dotenv import find_dotenv, load_dotenv

from src.data import make_folds
from src.features import make_feature_selection
from src.model import make_evaluation
from src.visualization import make_plots

if __name__ == '__main__':
    # Project path
    project_dir = str(Path(__file__).resolve().parents[1])
    # Find data.env automatically by walking up directories until it's found
    dotenv_path = find_dotenv(filename=join('parameters','pipeline_modules.env'))
    # Load up the entries as environment variables
    load_dotenv(dotenv_path)
    #
    dotenv_path = find_dotenv(filename=join('parameters','modules_parameters.env'))
    load_dotenv(dotenv_path)
    
    run_make_folds = environ.get('RUN_MAKE_FOLDS')
    run_fs_baselines = environ.get('RUN_FEATURE_SELECTION_BASELINES')
    run_model_train = environ.get('RUN_MODEL_TRAIN')
    run_model_predict = environ.get('RUN_MODEL_PREDICT')
    run_general_plots = environ.get('RUN_GENERAL_PLOTS')
    run_map_plots = environ.get('RUN_MAP_PLOTS')
    run_shap_plots = environ.get('RUN_SHAP_PLOTS')
    #
    ds_fold = make_folds.run(run_make_folds)
    make_feature_selection.run(run_fs_baselines, ds_fold)
    make_evaluation.run(run_model_train, run_model_predict, ds_fold)
    make_plots.run(run_general_plots, run_map_plots, run_shap_plots, ds_fold)
    
    
    
