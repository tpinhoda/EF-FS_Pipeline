import coloredlogs,  logging
coloredlogs.install()
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

import pickle
import shap
import pandas as pd
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from os import listdir
from os.path import join
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages
from src.utils import utils


def load_model(filepath):
    return pickle.load(open(filepath, 'rb'))


def generate_shap_plot(fold_name, model, x, pdf_pages):
    shap.initjs()
    f = plt.figure()
    explainerModel = shap.KernelExplainer(model.predict, x)
    shap_values_Model = explainerModel.shap_values(x)
    shap.summary_plot(shap_values_Model, x, show=False)
    plt.tight_layout()
    pdf_pages.savefig()
    return 0


def make_data(fold_path, target_col, filename):
    data = pd.read_csv(join(fold_path, filename), index_col='GEO_Cod_Municipio', low_memory=False)
    x, y = utils.split_data(data, target_col)
    return x, y

def run(folds_filepath, models_path, exp_filepath, map_plots, target_col, index_col):
    logger = logging.getLogger(__name__)
    fs_methods = [method for method in listdir(models_path)]
    folds_names = [fold_name for fold_name in listdir(folds_filepath)]
    logger.info('Generating Map Plots.')
    for fs_method in fs_methods:
        selected_features = utils.get_features_from_file(join(exp_filepath, 'features_selected', fs_method + '.json'))
        pdf_pages = PdfPages(join(map_plots, fs_method + '.pdf'))
        for fold_name in tqdm(folds_names):
            model = load_model(join(models_path, fs_method, fold_name  + '.sav'))
            x_test, y_test = make_data(join(folds_filepath, fold_name), target_col, 'test.csv')
            x_test = utils.filter_by_selected_features(x_test, selected_features)
            generate_shap_plot(fold_name, model, x_test, pdf_pages)
        pdf_pages.close()
    