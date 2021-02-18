import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.stats.morestats import _add_axis_labels_title
import shap
import seaborn as  sns
import mlflow
import json
import random
from sklearn.metrics import recall_score
from os import environ, listdir
from os.path import join
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import weightedtau, kendalltau, spearmanr, rankdata
from math import sqrt
from statistics import mean, stdev
from regression import LinearRegressionUsingGD
from optimizer import CustomLinearModel
from gradient_descent import gradient_descent
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import lightgbm


def residual_plot(model, x, y_true, sae, wkendall, meshblock):
    y_pred = model.predict(x)
    y_pred = pd.Series(y_pred, index=x.index, name='y_pred')
    
    rank_pred = pd.Series(rankdata(y_pred), index=x.index, name='rank_pred').astype('int64')
    rank_true = pd.Series(rankdata(y_true), index=x.index, name='rank_true').astype('int64')
    #OldRange = (y_pred.max() - y_pred.min())
    #NewRange = (y_true.max() - y_true.min())
    #y_pred = (((y_pred - y_pred.min()) * NewRange)/OldRange) + y_true.min()

    true_map = meshblock.merge(y_true, on='Cod_ap', how='left').copy()
    true_map = true_map.merge(rank_true, on='Cod_ap', how='left')
    true_map.dropna(axis=0, inplace=True)
    pred_map = meshblock.merge(y_pred, on='Cod_ap', how='left').copy()
    pred_map = pred_map.merge(rank_pred, on='Cod_ap', how='left')
    pred_map.dropna(axis=0, inplace=True)

    fig, ax = plt.subplots(2, 2)

    #ax[0][1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #ax[1][1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    ax[0][0].set_xticks([]) 
    ax[0][0].set_yticks([])
    ax[0][1].set_xticks([]) 
    ax[0][1].set_yticks([]) 
    box = ax[0][1].get_position()
    ax[0][1].set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    ax[0][0].set_title('True Distribution')
    ax[0][1].set_title('True Rank')
    true_map.plot(column='LISA', ax=ax[0][0], legend=True, cmap='RdYlBu')
    true_map.plot(column='rank_true', ax=ax[0][1], label='rank_pred', legend=True, cmap='RdYlBu')
    ax[1][0].set_xticks([]) 
    ax[1][0].set_yticks([]) 
    ax[1][1].set_xticks([]) 
    ax[1][1].set_yticks([]) 
    box = ax[1][1].get_position()
    ax[1][1].set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    ax[1][0].set_title('Predicted Distribution')
    ax[1][1].set_title('Predicted Rank')
    
  
    # Text
    textstr = '\n'.join((
    r'$\mathrm{SAE}=%.2f$' % (sae, ),
    r'$Kendall=%.2f$' % (wkendall, )))
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)

    # place a text box in upper left in axes coords
    ax[1][0].text(0.05, 0.95, textstr, transform=ax[1][0].transAxes, fontsize=7,
        verticalalignment='top', bbox=props)   
    pred_map.plot(column='y_pred', ax=ax[1][0], legend=True, cmap='RdYlBu')
    
  
    pred_map.plot(column='rank_pred', ax=ax[1][1], label=pred_map['rank_pred'].values, legend=True, cmap='RdYlBu')

    
    
    
    #fig.tight_layout()
    pp.savefig()
 


def calculate_mse(model, x, y_true):
    y_pred = pd.DataFrame(model.predict(x), index=x.index)
    OldRange = (y_pred[y_pred.columns[0]].max() - y_pred[y_pred.columns[0]].min())
    NewRange = (y_true[y_true.columns[0]].max() - y_true[y_true.columns[0]].min())
    y_pred = (((y_pred[y_pred.columns[0]] - y_pred[y_pred.columns[0]].min()) * NewRange)/OldRange) + y_true[y_true.columns[0]].min()

    #mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
    mse =  abs(y_pred - y_true[y_true.columns[0]]).sum()
    return mse


def calculate_wkendall(model, x, y_true):
    y_pred = model.predict(x)
    sorted_rank = rankdata(-y_pred, method='ordinal')
    sorted_rank = pd.DataFrame(sorted_rank)
    rank_votes = rankdata(-y_true, method='ordinal')
    rank_votes = pd.DataFrame(rank_votes)
    tau, _ = weightedtau(rank_votes, sorted_rank)
    #tau, _ = weightedtau(pd.DataFrame(y_true), pd.DataFrame(y_pred))
    return tau


def calculate_kendall(model, x, y_true):
    y_pred = model.predict(x)
    sorted_rank = rankdata(-y_pred)
    sorted_rank = pd.DataFrame(sorted_rank)
    rank_votes = rankdata(-y_true)
    rank_votes = pd.DataFrame(rank_votes)
    #tau, _ = kendalltau(rank_votes, sorted_rank)
    tau, _ = kendalltau(pd.DataFrame(y_true), pd.DataFrame(y_pred))
   
    return tau


def calculate_spearman(model, x, y_true):
    y_pred = model.predict(x)
    sorted_rank = rankdata(-y_pred, method='ordinal')
    sorted_rank = pd.DataFrame(sorted_rank)
    rank_votes = rankdata(-y_true, method='ordinal')
    rank_votes = pd.DataFrame(rank_votes)
    #ro, _ = spearmanr(rank_votes, sorted_rank)
    ro, _ = spearmanr(y_true, pd.DataFrame(y_pred))
    return ro


def calculate_recall(model, x, y_true, perc):
    target_col = 'LISA'
    y_pred = model.predict(x)
    y_pred = rankdata(-y_pred, method='ordinal') < max(perc*len(y_pred)/100, 3.1)
    y_true = rankdata(-y_true[target_col], method='ordinal') < max(perc*len(y_true)/100, 3.1)

    y_true = [1 if y else 0 for y in y_true]
    y_pred = [1 if pred else 0 for pred in y_pred]
    results = recall_score(y_true, y_pred, pos_label=1)
    return results


def get_pos1(model, x, y_true):
    y_pred = model.predict(x)
    idx_major = y_true[y_true['interest']=='major'].index.values
    sorted_rank = rankdata(y_pred)
    sorted_rank = pd.DataFrame(sorted_rank)
    rank_votes = rankdata(y_true['LISA'])
    rank_votes = pd.DataFrame(rank_votes)
    
    if y_true.loc[idx_major[0],'LISA'] > 0:
        pred_pos_1 = sorted_rank[sorted_rank[0] == len(y_true)].index.values
        if rank_votes.loc[pred_pos_1].values[0][0] == len(y_true):
            return 1
        else:
            return 0
    else:
        pred_pos_1 = sorted_rank[sorted_rank[0] == 1].index.values
        if rank_votes.loc[pred_pos_1].values[0][0] == 1:
            return 1
        else:
            return 0
    


def get_rep_folds(list_folds, k, rep):
    repetition_cv = dict()
    for i in range(rep):
        #repetition_cv[str(i)] = random.sample(list_folds, k)
        repetition_cv[str(i)] = list_folds
        list_folds = [f for f in list_folds if f not in repetition_cv[str(i)]]
    return repetition_cv


def split_data(data, levels):
    data.set_index('Cod_ap', inplace=True)
    print(data.columns.values)
    code_cols = ['Cod_Setor', 'Cod_ap', 'Cod_Grande_Regiao', 'Nome_Grande_Regiao', 'Cod_UF',
                 'Nome_UF', 'Cod_Meso', 'Nome_Meso', 'Nome_Micro', 'Cod_Micro'
                 'Cod_RM', 'Nome_RM', 'Cod_Municipio', 'Nome_Municipio',
                 'Cod_Distrito', 'Nome_Distrito', 'Cod_Subdistrito',
                 'Nome_Subdistrito', 'Cod_Bairro', 'Nome_Bairro', 'Cod_Country']
   # output_space = ['{}_STRANGENESS'.format(level.upper())
   #                 for level in levels]
    output_space = ['LISA']
   
    discret_output_space = ['quantile_{}_STRANGENESS'.format(level.upper())
                            for level in levels]
    out_all = output_space + discret_output_space + code_cols
    input_space = [col for col in data.columns.values if col not in out_all]
    return data[input_space], data[output_space + ['vote_shares', 'interest']]


def make_train_test(path, fold, levels):
    train = pd.read_csv(join(path, fold, 'train.csv'), low_memory=False)
    test = pd.read_csv(join(path, fold, 'test.csv'), low_memory=False)
 
    x_train, y_train = split_data(train, levels)
    x_test, y_test = split_data(test, levels)
    return x_train, y_train, x_test, y_test


def transform_data(data, feature_selected, scale=False):
    f_list = []
    for f in feature_selected:
        if scale is True:
            data[f] = ((data[f] - data[f].min()) / (data[f].max() - data[f].min())) * (feature_selected[f]['max'] - feature_selected[f]['min']) + feature_selected[f]['min']
            data[f].fillna(feature_selected[f]['min'], inplace=True)
        
        f_list.append(f)
    return data[f_list]


def create_eval_dict(recall_percs):
    eval_dict = {'n_features': [],
                 'n_test': [],
                 'r2': [],
                 'mse': [],
                 'kendall': [],
                 'wkendall': [],
                 'spearmanr': [],
                 'pos_1': [],
                 }
    for perc in recall_percs:
        eval_dict['recall_{}'.format(str(perc))] = []
    return eval_dict


def spatial_cross_validation(path_results, model, features_selected, path, repetition_cv, method, meshblock_path):
    recall_percs = [1, 10, 20, 30, 40, 50]
    rep_metrics = create_eval_dict(recall_percs)
    meshblock = gpd.read_file(meshblock_path)
    
    for cv in repetition_cv:
        cv_metrics = create_eval_dict(recall_percs)
        
        repetition_cv[cv] = list(map(str, (sorted(list(map(int, repetition_cv[cv]))))))
        print(repetition_cv[cv])
        for fold in tqdm(repetition_cv[cv]):
            
            x_train, y_train, x_test, y_test = make_train_test(path, fold, ['NM_COUNTRY'])
            x_train = transform_data(x_train, features_selected, False)
            x_test = transform_data(x_test, features_selected, False)
            
          #  y_train['LISA'] = y_train['LISA']* -1
          #  y_test['LISA'] = y_test['LISA']* -1 
          #  x_test.drop('labels', axis=1, inplace=True)
          #  y_test.drop('labels', axis=1, inplace=True)
            # For Regression
          #  x_train.drop('labels', axis=1, inplace=True)
          #  y_train.drop('labels', axis=1, inplace=True)
        
            model.fit(x_train, y_train['LISA'])
            # ===========================
            #shap_plot(model, x_test, y_test['vote_shares'], fold)
            cv_metrics['n_features'].append(len(features_selected))
            cv_metrics['mse'].append(calculate_mse(model, x_test, y_test[['LISA']]))
            cv_metrics['r2'].append(0)
            cv_metrics['n_test'].append(len(y_test))
            cv_metrics['kendall'].append(calculate_kendall(model, x_test, y_test[['LISA']]))
            cv_metrics['wkendall'].append(calculate_wkendall(model, x_test, y_test[['LISA']]))
            cv_metrics['spearmanr'].append(calculate_spearman(model, x_test, y_test[['LISA']]))
            
            cv_metrics['pos_1'].append(get_pos1(model, x_test, y_test[['LISA', 'interest']]))
            print(' size: {} - SAE: {} - Kendall: {}'.format(len(x_test),cv_metrics['mse'][-1], cv_metrics['kendall'][-1]))
            residual_plot(model, x_test, y_test['LISA'],cv_metrics['mse'][-1], cv_metrics['kendall'][-1], meshblock)   
            for perc in recall_percs:
                cv_metrics['recall_{}'.format(perc)].append(calculate_recall(model, x_test, y_test, perc))

    cv_metrics = pd.DataFrame(cv_metrics)
    cv_metrics.to_csv(join(path_results, method.split('.')[0]+'.csv'), index=False)

def custom_asymmetric_train(y_true, y_pred):
    rank_true = rankdata(-1*y_true)
    rank_pred = rankdata(-1*y_pred)
    
    residual = abs(rank_true - rank_pred).astype("float")
    grad = residual
    hess = np.where(residual<0, 2, 2)
    return grad, hess

def custom_asymmetric_valid(y_true, y_pred):
    residual = abs(y_true - y_pred).astype("float")
    return "custom_asymmetric_eval", np.sum(residual), False


if __name__ == '__main__':
    # Find data.env automatically by walking up directories until it's found
    dotenv_path = find_dotenv(filename='data.env')
    # Load up the entries as environment variables
    load_dotenv(dotenv_path)
    # Get dataset parameter
    region = environ.get('REGION_NAME')
    tse_year = str(environ.get('ELECTION_YEAR'))
    tse_office = environ.get('POLITICAL_OFFICE')
    tse_turn = str(environ.get('ELECTION_TURN'))
    tse_per = environ.get('PER')
    candidates = environ.get('CANDIDATES').split(',')
    ibge_year = str(environ.get('CENSUS_YEAR'))
    ibge_aggr = environ.get('CENSUS_AGGR_LEVEL')
    fold_group = environ.get('N_FOLDS')
    # Get data root path
    data_dir = environ.get('ROOT_DATA')
    input_dir = environ.get('INPUT_DATA')
    # Get mesh blocks
    path_meshblocks = input_dir + environ.get('MESHBLOCKS')
    processed_path_meshblocks = path_meshblocks.format(region, ibge_year, 'processed')
    input_filepath_meshblock = join(processed_path_meshblocks, ibge_aggr, 'shapefiles', region+'.shp')
    # Get census results path
    folds_path = data_dir + environ.get('FOLDS_PATH')
    folds_path = folds_path.format(
        region, tse_year, ibge_year, tse_office, tse_turn, ibge_aggr, tse_per, fold_group)
    # Results path
    results_path = data_dir + environ.get('RESULTS_PATH')
    results_path = results_path.format(
        region, tse_year, ibge_year, tse_office, tse_turn, ibge_aggr, tse_per)
    path_fs = join(results_path, 'selected_features')

    results_path = join(results_path, 'linear_regression', 'by_folds_{}'.format(fold_group))
    # Creating results folders
    Path(results_path).mkdir(parents=True, exist_ok=True)
    # Project path
    project_dir = str(Path(__file__).resolve().parents[2])
    # Get election results path
    path = 'file:' + project_dir + environ.get("LOGS").format(region)
    # Set mflow log dir
    mlflow.set_tracking_uri(path)
    try:
        mlflow.create_experiment('Hierarchical Feature Selection')
    except:
        mlflow.set_experiment('Hierarchical Feature Selection')

    folds = [fold for fold in listdir(folds_path)]
    rep_cv = get_rep_folds(folds, k=66, rep=1)
    method_folder = 'baselines'
    proposed_group = ''
    fs_methods = [method for method in listdir(join(path_fs, method_folder, proposed_group))]
    l_results = []
    for file_features in fs_methods:
        print(file_features)
        pp = PdfPages(join(results_path, method_folder+'_residual_maps', '{}.pdf'.format(file_features.split('.')[0])))
        with open(join(path_fs, method_folder, proposed_group, file_features)) as f:
            selected_features = json.load(f)
        #model = LinearRegressionUsingGD()
        #model = lightgbm.LGBMRegressor()
        model = LinearRegression()
        #model.set_params(**{'objective': custom_asymmetric_train}, metrics=["mse", 'mae'])

        spatial_cross_validation(join(results_path, method_folder, proposed_group), model,
                                 selected_features, folds_path, rep_cv, file_features, input_filepath_meshblock)
        pp.close()