import logging
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

def single_shap_plot(fold_name, plot_path, model, j, x):
    shap.initjs()
    f = plt.figure()
    f.suptitle('Fold: {}'.format(fold_name), fontsize=16)
    explainerModel = shap.TreeExplainer(model)
    shap_values_Model = explainerModel.shap_values(x)
    fp = shap.force_plot(explainerModel.expected_value, shap_values_Model[len(x)-1], x.loc[[j]], show=False)
    shap.save_html(join(plot_path, fold_name + '.html'), fp)
    #plt.tight_layout()
    #plt.savefig(join(plot_path, fold_name + '.html'))
    #plt.close()


def generate_shap_plot(fold_name, model, x, pdf_pages):
    shap.initjs()
    f = plt.figure()
    f.suptitle('Fold: {}'.format(fold_name), fontsize=16)
    explainerModel = shap.KernelExplainer(model.predict, x)
    shap_values_Model = explainerModel.shap_values(x)
    shap.summary_plot(shap_values_Model, x, show=False)
    plt.tight_layout()
    pdf_pages.savefig()
    plt.close('all')



def make_data(fold_path, target_col, filename):
    data = pd.read_csv(join(fold_path, filename), index_col='GEO_Cod_Municipio', low_memory=False)
    x, y = utils.split_data(data, target_col)
    return x, y

def run(folds_filepath, models_path, exp_filepath, map_plots, target_col, index_col):
    logger_name = 'Visualization'
    logger = logging.getLogger(logger_name)
    fs_methods = [method for method in listdir(models_path)]
    folds_names = [fold_name for fold_name in listdir(folds_filepath)]
    logger.info('Generating Map Plots.')
    for fs_method in fs_methods:
        selected_features = utils.get_features_from_file(join(exp_filepath, 'features_selected', fs_method + '.json'))
        #pdf_pages = PdfPages(join(map_plots, fs_method + '.pdf'))
        plot_path = utils.create_folder(map_plots, fs_method, logger_name)
        for fold_name in tqdm(folds_names, desc='Generating shap plots', leave=False):
            model = load_model(join(models_path, fs_method, fold_name  + '.sav'))
            x_test, y_test = make_data(join(folds_filepath, fold_name), target_col, 'test.csv')
            center_neighbor = x_test['center_neighbor']
            center = x_test.index[x_test['center_neighbor'] == 'center'].tolist()
            if len(center) > 1:
                x_test = x_test.iloc[0:-1,:]

            x_test = utils.filter_by_selected_features(x_test, selected_features)
            
            #generate_shap_plot(fold_name, model, x_test, pdf_pages)
            single_shap_plot(fold_name, plot_path, model, center[0], x_test)
        #pdf_pages.close()
    