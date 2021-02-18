from src.model.make_prediction import calculate_rmse
import coloredlogs,  logging
coloredlogs.install()
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

import pickle
import pandas as pd
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import rankdata, kendalltau
from os import listdir
from os.path import join
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages
from src.utils import utils


def load_model(filepath):
    return pickle.load(open(filepath, 'rb'))


def calculate_kendall(model, x, y_true):
    y_pred = model.predict(x)
    tau, _ = kendalltau(y_true, y_pred)
    return tau


def calculate_sae(model, x, y_true):
    y_pred = model.predict(x)
    OldRange = (y_pred.max() - y_pred.min())
    NewRange = (y_true.max() - y_true.min())
    y_pred = (((y_pred - y_pred.min()) * NewRange)/OldRange) + y_true.min()
    sae =  abs(y_pred - y_true).sum()
    return sae

def merge_meshblock_results(meshblock, y, rank, key):
    map_data = meshblock.merge(y, on=key, how='left').copy()
    map_data = map_data.merge(rank, on=key, how='left')
    map_data.dropna(axis=0, inplace=True)
    return map_data
    
def create_index_col_meshblock(meshblock, index_col):
    if index_col == 'GEO_Cod_Municipio':
        new_meshblock = meshblock.rename(columns = {'CD_GEOCMU': 'GEO_Cod_Municipio'})
        key = 'GEO_Cod_Municipio' 
        new_meshblock[key] = new_meshblock[key].astype('int64')
    elif index_col == 'GEO_Cod_ap':
        new_meshblock = meshblock.rename(columns = {'Cod_ap': 'GEO_Cod_ap'})
        key = 'GEO_Cod_ap'
        new_meshblock[key] = new_meshblock[key].astype('int64')
    return new_meshblock, key


def plot_map(map_data, ax, row, col, target_col, title, cmap, text=False, sae=0, kendall=0):
    ax[row][col].set_title(title)
    ax[row][col].set_xticks([]) 
    ax[row][col].set_yticks([])
    if text:
        textstr = '\n'.join((
                  r'$\mathrm{SAE}=%.2f$' % (sae, ),
                  r'$Kendall=%.2f$' % (kendall, )))
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)
        # place a text box in upper left in axes coords
        ax[row][col].text(0.05, 0.95, textstr, transform=ax[row][col].transAxes, fontsize=7,
            verticalalignment='top', bbox=props)  
    map_data.plot(column=target_col, ax=ax[row][col], legend=True, cmap=cmap)
    
    
def generate_map_plot(fold_name, model, x, y_true, meshblock, index_col, target_col, pdf_pages):
    # Making predictions
    y_pred = model.predict(x)
    y_pred = pd.Series(y_pred, index=x.index, name='y_pred')
    # Making rankings
    rank_pred = pd.Series(rankdata(y_pred), index=x.index, name='rank_pred').astype('int64')
    rank_true = pd.Series(rankdata(y_true), index=x.index, name='rank_true').astype('int64')
    # Using the same index col as original data
    map_data, key = create_index_col_meshblock(meshblock, index_col)
    # Adding y and rank to the meshblock
    true_map = merge_meshblock_results(map_data, y_true, rank_true, key)
    pred_map = merge_meshblock_results(map_data, y_pred, rank_pred, key)
    # Calculate text metrics
    sae = calculate_sae(model, x,y_true)
    kendall = calculate_kendall(model, x,y_true)
    fig, ax = plt.subplots(2, 2)
    fig.suptitle('Fold: {}'.format(fold_name), fontsize=16)
    cmap = 'autumn'
    plot_map(true_map, ax, 0, 0, target_col, 'True Distribution', cmap)
    plot_map(pred_map, ax, 0, 1, 'y_pred', 'Predicted Distribution', cmap, True, sae, kendall)
    cmap = 'RdYlBu'
    plot_map(true_map, ax, 1, 0, 'rank_true', 'True Rank', cmap)
    plot_map(pred_map, ax, 1, 1, 'rank_pred', 'Predicted Rank', cmap)
    pdf_pages.savefig()
    
def make_data(fold_path, target_col, filename):
    data = pd.read_csv(join(fold_path, filename), index_col='GEO_Cod_Municipio', low_memory=False)
    x, y = utils.split_data(data, target_col)
    return x, y

def run(folds_filepath, models_path, exp_filepath, map_plots, meshblock_filepath, target_col, index_col):
    logger = logging.getLogger(__name__)
    fs_methods = [method for method in listdir(models_path)]
    folds_names = [fold_name for fold_name in listdir(folds_filepath)]
    logger.info('Generating Map Plots.')
    meshblock = gpd.read_file(meshblock_filepath)
    for fs_method in fs_methods:
        selected_features = utils.get_features_from_file(join(exp_filepath, 'features_selected', fs_method + '.json'))
        pdf_pages = PdfPages(join(map_plots, fs_method + '.pdf'))
        for fold_name in tqdm(folds_names):
            model = load_model(join(models_path, fs_method, fold_name  + '.sav'))
            x_test, y_test = make_data(join(folds_filepath, fold_name), target_col, 'test.csv')
            x_test = utils.filter_by_selected_features(x_test, selected_features)
            generate_map_plot(fold_name, model, x_test, y_test, meshblock,  index_col, target_col, pdf_pages)
        pdf_pages.close()
        
    