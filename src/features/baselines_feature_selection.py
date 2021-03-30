import logging
import pandas as pd
import numpy as np
import json
import random
from os import environ
from os.path import join
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm
from scipy.stats import weightedtau

import weka.core.jvm as jvm
from weka.attribute_selection import ASEvaluation, ASSearch, AttributeSelection
from weka.core.dataset import create_instances_from_matrices

from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

from src.utils import utils


def select_all_features(data_path, results_path):
    data = pd.read_csv(data_path, low_memory=False)
    x  = utils.get_descriptive_attributes(data)
    features = x.columns.values.tolist()
    json_features = {'selected_features': features}
    with open(join(results_path, 'topline.json'), "w") as fp:
        json.dump(json_features, fp, indent=4)


def select_worst_case_sklearn(data_path, results_path, n_features, target_col):
    logger_name = 'FS Baselines'
    logger = logging.getLogger(logger_name)
    data = pd.read_csv(data_path, low_memory=False)
    x = utils.get_descriptive_attributes(data)
    y = data[target_col]
    for method in ['regression', 'mi']:
        logger.info('Selecting {} worst features based on {}.'.format(n_features, method))
        if method == 'regression':
            fs = SelectKBest(score_func=f_regression, k='all')
        if method == 'mutual_information':
            fs = SelectKBest(score_func=mutual_info_regression, k='all')
        fs.fit(x, y)
        worst_idx = np.argsort(fs.scores_)[:n_features-1]
        features = x.columns[worst_idx].values.tolist()
        json_features = {'selected_features': features}
        with open(join(results_path, 'worst_{}.json'.format(method)), "w") as fp:
            json.dump(json_features, fp, indent=4)
        

def select_random_features_perc(data_path, results_path, perc):
    data = pd.read_csv(data_path, low_memory=False)
    x = utils.get_descriptive_attributes(data)
    features = x.columns.values.tolist()
    k = int(perc*len(features)/100)
    features = random.sample(features, k=k)
    json_features = {'selected_features': features}
    with open(join(results_path, 'random_{}%.json'.format(str(perc))), "w") as fp:
        json.dump(json_features, fp, indent=4)


def select_random_features_number(data_path, results_path, n_features):
    data = pd.read_csv(data_path, low_memory=False)
    x = utils.get_descriptive_attributes(data)
    features = x.columns.values.tolist()
    features = random.sample(features, k=n_features)
    json_features = {'selected_features': features}
    with open(join(results_path, 'n_rand.json'), "w") as fp:
        json.dump(json_features, fp, indent=4)


def correlation_methods(data_path, results_path, n_features, target_col, worst=False):
    logger_name = 'FS Baselines'
    logger = logging.getLogger(logger_name)
    
    def wkendall(x, y):
        tau, _ = weightedtau(x, y)
        return tau

    data = pd.read_csv(data_path, low_memory=False)
    x = utils.get_descriptive_attributes(data)
    y = data[target_col]
    for method in ['pearson', 'kendall', 'spearman']:
        logger.info('Selecting {} features based on {}.'.format(n_features, method))
        if method != 'wkendall':
            cor = x.corrwith(y, axis=0, method=method)
        else:
            cor = x.corrwith(y, axis=0, method=wkendall)
        # Selecting highly correlated features
        if worst:
            cor = cor.abs()
            features = cor.nsmallest(n_features).index.values.tolist()
            json_features = {'selected_features': features}
            with open(join(results_path, 'worst_{}.json'.format(method)), "w") as fp:
                json.dump(json_features, fp, indent=4)
        else:    
            features = cor.nlargest(n_features).index.values.tolist()
            json_features = {'selected_features': features}
            with open(join(results_path, '{}.json'.format(method)), "w") as fp:
                json.dump(json_features, fp, indent=4)


def sklearn_methods(data_path, results_path, n_features, target_col, worst=False):
    logger_name = 'FS Baselines'
    logger = logging.getLogger(logger_name)
    data = pd.read_csv(data_path, low_memory=False)
    x = utils.get_descriptive_attributes(data)
    y = data[target_col]
    for method in ['regression', 'mi']:
        logger.info('Selecting {} features based on {}.'.format(n_features, method))
        if method == 'regression':
            fs = SelectKBest(score_func=f_regression, k=n_features)
        if method == 'mutual_information':
            fs = SelectKBest(score_func=mutual_info_regression, k=n_features)
        fs.fit(x, y)
        cols = fs.get_support(indices=True)
        features = x.iloc[:,cols].columns.values.tolist()
        json_features = {'selected_features': features}
        with open(join(results_path, '{}.json'.format(method)), "w") as fp:
            json.dump(json_features, fp, indent=4)
    

def weka_methods(data_path, results_path, n_features, target_col):
    logger_name = 'FS Baselines'
    logger = logging.getLogger(logger_name)
    logger.info('Starting JVM.')
    logger.warning('Some warnings may appear.')
    jvm.start()
    data = pd.read_csv(data_path, low_memory=False)
    x = utils.get_descriptive_attributes(data)
    x_fs = x.copy()
    x_fs['target'] = data[target_col]
    x_fs = create_instances_from_matrices(x_fs.to_numpy())
    x_fs.class_is_last()
    for method in ['cfs', 'rrelieff']:
        if method == 'rrelieff':
            logger.info('Selecting {} features based on {}.'.format(n_features, method))
            search = ASSearch(classname="weka.attributeSelection.Ranker", options=[
                              '-T', '-1.7976931348623157E308', '-N', str(n_features-1)])
            evaluator = ASEvaluation(classname="weka.attributeSelection.ReliefFAttributeEval",
                                     options=["-M", "1", "-D", "1", '-K', '10'])
        else:
            logger.info('Selecting features based on {}.'.format(method))
            search = ASSearch(classname="weka.attributeSelection.BestFirst", options=["-D", "1", "-N", "5"])
            evaluator = ASEvaluation(classname="weka.attributeSelection.CfsSubsetEval", options=["-P", "1", "-E", "1"])
        attsel = AttributeSelection()
        attsel.search(search)
        attsel.evaluator(evaluator)
        attsel.select_attributes(x_fs)
        index_fs = [i - 1 for i in attsel.selected_attributes]
        features = x.columns.values[index_fs].tolist()
        json_features = {'selected_features': features}
        with open(join(results_path, '{}.json'.format(method)), "w") as fp:
            json.dump(json_features, fp, indent=4)
        if n_features == -1:
            n_features = len(features)
    jvm.stop()
    return n_features



def run(run_fs_baselines, ds_fold):
    logger_name = 'FS Baselines'
    logger = logging.getLogger(logger_name)
    if run_fs_baselines == 'True':
        # Find data.env automatically by walking up directories until it's found
        dotenv_path = find_dotenv(filename=join('parameters','feature_selection.env'))
        # Load up the entries as environment variables
        load_dotenv(dotenv_path)
        # Get dataset parameter
        input_filepath = environ.get('INPUT_DATASET')
        output_filepath = environ.get('OUTPUT_PATH')
        output_filepath = join(output_filepath,ds_fold)
        # Get data fold parameters
        type_folds = environ.get('TYPE_FOLDS')
        target_col = environ.get('TARGET')
        n_features = int(environ.get('FILTERING_N_FEATURES'))
        random_perc = int(environ.get('RANDOM_PERC'))
        output_filepath = utils.get_fold_type_folder_path(type_folds, output_filepath, logger_name)
        if type_folds == 'CN':
            n_neighbors = environ.get('C_N_NEIGHBORS')
            filter_train = environ.get('FILTER_TRAIN')
            center_candidate = environ.get('CENTER_CANDIDATE')
            folder_name = center_candidate + '_' + 'N'+n_neighbors+'_FT_'+filter_train
            output_filepath = join(output_filepath, folder_name)
            if filter_train == 'True':
                input_filepath = join(output_filepath, 'filtered_data.csv')

        output_filepath = utils.create_folder(output_filepath, 'experiments', logger_name)
        if n_features == -1:
            output_filepath = utils.create_folder(output_filepath, 'TG_{}_FN_CFS_RP_{}'.format(target_col, str(random_perc)), logger_name)
        else:
            output_filepath = utils.create_folder(output_filepath, 'TG_{}_FN_{}_RP_{}'.format(target_col, str(n_features), str(random_perc)),logger_name)
        output_filepath = utils.create_folder(output_filepath, 'features_selected', logger_name)
        # =============================================


        logger.info('Selecting all features.')
        select_all_features(input_filepath, output_filepath)
        logger.info('Selecting features based on weka merhods: [RReliefF, CFS].')
        n_features = weka_methods(input_filepath, output_filepath, n_features, target_col)
        logger.info('Selecting features based on correlation: [pearson, kendall, spearman].')
        correlation_methods(input_filepath, output_filepath, n_features, target_col)
        logger.info('Selecting features based on sklearn methods: [regression, mutual information].')
        sklearn_methods(input_filepath, output_filepath, n_features, target_col)
        logger.info('Selecting worst features sklearn.')
        select_worst_case_sklearn(input_filepath, output_filepath, n_features, target_col)
        logger.info('Selecting worst features correlation.')
        correlation_methods(input_filepath, output_filepath, n_features, target_col, worst=True)
        
        # logger.info('Selecting {} random features.'.format(str(n_features)))
        # select_random_features_number(input_filepath, output_filepath, n_features)
        # logger.info('Selecting {}% random features.'.format(str(random_perc)))
        # select_random_features_perc(input_filepath, output_filepath, random_perc)
    else:
        logger.warning('Not running feature selection baselines.')
        
    