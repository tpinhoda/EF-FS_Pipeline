from src.model.make_prediction import calculate_rmse
import logging
import pickle
import pandas as pd
import numpy as np
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import rankdata, kendalltau
from os import listdir
from os.path import join
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages
from src.utils import utils

plt.rc('legend',fontsize=4)
plt.rc('figure',max_open_warning=100)


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


def plot_map(map_data, ax, row, col, target_col, title, cmap, text=False, move_legend=False, sae=0, kendall=0):
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
        ax[row][col].text(0.05, 0.95, textstr, transform=
                          ax[row][col].transAxes, fontsize=7,
            verticalalignment='top', bbox=props)
    if 'Distribution' in title:
        map_data.plot(column=target_col, ax=ax[row][col], legend=True, cmap=cmap, vmin=0, vmax=100)
    if 'Rank' in title:
        map_data.plot(column=target_col, ax=ax[row][col], legend=True, cmap=cmap)
    if 'Win' in title:
        map_data.plot(column=target_col, ax=ax[row][col], legend=True, cmap=cmap)

       # legend = handles=ax[row][col].get_legend()
       # handles = [] if legend is None else legend.legendHandles
       # plt.legend(handles=handles, title='title', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    
def generate_map_plot(fold_name, model, x, y_true, meshblock, who_won, index_col, target_col, vote_shares, candidate, pdf_pages):
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
    # Adding who won and vote shares to meshblock 
    true_win_map = map_data.merge(who_won, on=key, how='left')
    true_win_map.dropna(axis=0, inplace=True)
    pred_map['won_pred'] = np.where(pred_map['y_pred'] < 50, 'Lost', 'Win')
    win_lost_type = pd.CategoricalDtype(categories=['Win', 'Lost'], ordered=False)
    pred_map['won_pred'] = pred_map['won_pred'].astype(win_lost_type)
    
    # Adding voteshares to meshblock
    vote_map = map_data.merge(vote_shares, on=key, how='left')
    vote_map.dropna(axis=0, inplace=True)
    # Calculate text metrics
    sae = calculate_sae(model, x,y_true)
    kendall = calculate_kendall(model, x,y_true)
    fig, ax = plt.subplots(3, 2)
    fig.suptitle('Fold: {}'.format(fold_name), fontsize=16)
    cmap = 'YlOrRd'
    plot_map(true_map, ax, 0, 0, target_col, 'True Distribution', cmap)
    plot_map(pred_map, ax, 0, 1, 'y_pred', 'Predicted Distribution', cmap, text=True, sae=sae, kendall=kendall)
    cmap = 'RdYlBu'
    plot_map(true_map, ax, 1, 0, 'rank_true', 'True Rank', cmap)
    plot_map(pred_map, ax, 1, 1, 'rank_pred', 'Predicted Rank', cmap)
    cmap='Paired'
    plot_map(true_win_map, ax, 2, 0, 'ELECTION_who_won', 'True Win Map', cmap, move_legend=True)
    cmap='Paired'
    plot_map(pred_map, ax, 2, 1, 'won_pred', 'Predicted Win Map', cmap, move_legend=True)
    pdf_pages.savefig(bbox_inches="tight")
    plt.close('all')
    
def make_data(fold_path, target_col, filename):
    data = pd.read_csv(join(fold_path, filename), index_col='GEO_Cod_Municipio', low_memory=False)
    x, y = utils.split_data(data, target_col)
    return x, y

def get_vote_shares(data, candidate):
    if candidate == "HADDAD":
        candidate_col = 'ELECTION_FERNANDO HADDAD'
    else:
        candidate_col = 'ELECTION_JAIR BOLSONARO'
    
    data[candidate_col+'(%)'] = data[candidate_col] / data['ELECTION_QT_COMPARECIMENTO']
    return candidate_col+'(%)', data[candidate_col+'(%)']
       
    

def run(folds_filepath, models_path, exp_filepath, map_plots, meshblock_filepath, target_col, index_col, center_candidate):
    logger_name = 'Visualization'
    logger = logging.getLogger(logger_name)
    fs_methods = [method for method in listdir(models_path)]
    folds_names = [fold_name for fold_name in listdir(folds_filepath)]
    logger.info('Generating Map Plots.')
    meshblock = gpd.read_file(meshblock_filepath)
    for fs_method in fs_methods:
        logger.info('Generating for: {}'.format(fs_method))
        selected_features = utils.get_features_from_file(join(exp_filepath, 'features_selected', fs_method + '.json'))
        pdf_pages = PdfPages(join(map_plots, fs_method + '.pdf'))
        for fold_name in tqdm(folds_names, desc='Plotting maps', leave=False):
            model = load_model(join(models_path, fs_method, fold_name  + '.sav'))
            x_test, y_test = make_data(join(folds_filepath, fold_name), target_col, 'test.csv')
            who_won = pd.Series(np.where(x_test['ELECTION_who_won'] == center_candidate, 'Win', 'Lost'), index=x_test.index, name='ELECTION_who_won')
            win_lost_type = pd.CategoricalDtype(categories=['Win', 'Lost'], ordered=False)
            who_won = who_won.astype(win_lost_type)
            _, candidate_vote_shares = get_vote_shares(x_test.copy(), center_candidate)
            x_test = utils.filter_by_selected_features(x_test, selected_features)
            generate_map_plot(fold_name, model, x_test, y_test, meshblock, who_won, index_col, target_col, candidate_vote_shares, center_candidate, pdf_pages)
        pdf_pages.close()
        
        
    