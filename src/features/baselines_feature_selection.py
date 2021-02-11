import coloredlogs,  logging
coloredlogs.install()
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

import pandas as pd
import json
import random
from os import environ, mkdir
from os.path import join
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm
from scipy.stats import weightedtau

import weka.core.jvm as jvm
from weka.attribute_selection import ASEvaluation, ASSearch, AttributeSelection
from weka.core.dataset import create_instances_from_matrices

def get_descriptive_attributes(data):
    census_cols = [c for c in data.columns if 'CENSUS' in c]
    idhm_cols = [c for c in data.columns if 'IDHM' in c]
    elections_cols = [c for c in data.columns if 'ELECTION' in c]
    
    elections_in_cols = [c for c in elections_cols if 'BOLSONARO' not in c]
    elections_in_cols = [c for c in elections_in_cols if 'HADDAD' not in c]
    elections_in_cols = [c for c in elections_in_cols if 'who_won' not in c]
    
    input_space = census_cols + idhm_cols + elections_in_cols
    return data[input_space]


def select_all_features(data_path, results_path):
    data = pd.read_csv(data_path, low_memory=False)
    x  = get_descriptive_attributes(data)
    features = x.columns.values.tolist()
    json_features = {'selected_features': features}
    with open(join(results_path, 'all_features.json'), "w") as fp:
        json.dump(json_features, fp, indent=4)


def select_random_features(data_path, results_path, perc):
    data = pd.read_csv(data_path, low_memory=False)
    x = get_descriptive_attributes(data)
    features = x.columns.values.tolist()
    k = int(perc*len(features)/100)
    features = random.sample(features, k=k)
    json_features = {'selected_features': features}
    with open(join(results_path, 'random_{}%.json'.format(str(perc))), "w") as fp:
        json.dump(json_features, fp, indent=4)


def filtering_method(data_path, results_path, n_features, target_col):
    logger = logging.getLogger(__name__)
    
    def wkendall(x, y):
        tau, _ = weightedtau(x, y)
        return tau

    data = pd.read_csv(data_path, low_memory=False)
    x = get_descriptive_attributes(data)
    y = data[target_col]
    for method in ['pearson', 'kendall', 'spearman', 'wkendall']:
        logger.info('Selecting {} features based on {}.'.format(n_features, method))
        if method != 'wkendall':
            cor = x.corrwith(y, axis=0, method=method)
        else:
            cor = x.corrwith(y, axis=0, method=wkendall)
        # Selecting highly correlated features
        features = cor.nlargest(n_features).index.values.tolist()
        json_features = {'selected_features': features}
        with open(join(results_path, '{}.json'.format(method)), "w") as fp:
            json.dump(json_features, fp, indent=4)


def weka_methods(data_path, results_path, n_features, target_col):
    logger = logging.getLogger(__name__)
    logger.info('Starting JVM, some warnings may appear.')
    jvm.start()
    data = pd.read_csv(data_path, low_memory=False)
    x = get_descriptive_attributes(data)
    x_fs = x.copy()
    x_fs['target'] = data[target_col]
    x_fs = create_instances_from_matrices(x_fs.to_numpy())
    x_fs.class_is_last()
    for method in ['csf', 'rrelieff']:
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

def get_folder_name(type_folds, output_filepath):
    logger = logging.getLogger(__name__)
    if type_folds == 'R':
        output_filepath = join(output_filepath, 'Regiao')
    elif type_folds == 'S':
        output_filepath = join(output_filepath, 'UF')
    elif type_folds == 'ME':
        output_filepath = join(output_filepath, 'Meso')
    elif type_folds == 'MI':
        output_filepath = join(output_filepath, 'Micro')
    elif type_folds == 'D':
        output_filepath = join(output_filepath, 'Distrito')
    elif type_folds == 'SD':
        output_filepath = join(output_filepath, 'Subdistrito')
    elif type_folds == 'CN':
        output_filepath = join(output_filepath, 'Changing_Neighborhood')
    else:
        output_filepath = None
        logger.info('Incorrect type fold option try: [R, S, ME, MI, D, SD, CN]')
        exit()
    return output_filepath

def create_folder(path, folder_name):
    logger = logging.getLogger(__name__)
    path = join(path, folder_name)
    try:
        mkdir(path)
        logger.info('Creating Folder: {}'.format(folder_name))
    except FileExistsError:
        logger.info('Entenring Folder: {}'.format(folder_name))
    return path

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    # Find data.env automatically by walking up directories until it's found
    dotenv_path = find_dotenv(filename=join('parameters','feature_selection.env'))
    # Load up the entries as environment variables
    load_dotenv(dotenv_path)
    # Get dataset parameter
    input_filepath = environ.get('INPUT_DATASET')
    output_filepath = environ.get('OUTPUT_PATH')
    # Get data fold parameters
    type_folds = environ.get('TYPE_FOLDS')
    target_col = environ.get('TARGET')
    n_features = int(environ.get('FILTERING_N_FEATURES'))
    random_perc = int(environ.get('RANDOM_PERC'))
    output_filepath = get_folder_name(type_folds, output_filepath)
    if type_folds == 'CN':
        n_neighbors = environ.get('C_N_NEIGHBORS')
        filter_train = environ.get('FILTER_TRAIN')
        center_candidate = environ.get('CENTER_CANDIDATE')
        folder_name = center_candidate + '_' + 'N'+n_neighbors+'_FT_'+filter_train
        output_filepath = join(output_filepath, folder_name)
        if filter_train == 'True':
            input_filepath = join(output_filepath, 'filtered_data.csv')

    output_filepath = create_folder(output_filepath, 'experiments')
    if n_features == -1:
        output_filepath = create_folder(output_filepath, 'TG_{}_FN_CFS_RP_{}'.format(target_col, str(random_perc)))
    else:
        output_filepath = create_folder(output_filepath, 'TG_{}_FN_{}_RP_{}'.format(target_col, str(n_features), str(random_perc)))
    output_filepath = create_folder(output_filepath, 'features_selected')
    # =============================================


    logger.info('Selecting all features.')
    select_all_features(input_filepath, output_filepath)
    logger.info('Selecting {}% random features.'.format(str(random_perc)))
    select_random_features(input_filepath, output_filepath, random_perc)
    logger.info('Selecting features based on weka merhods: [RReliefF, CFS].')
    n_features = weka_methods(input_filepath, output_filepath, n_features, target_col)
    logger.info('Selecting features based on filtering: [pearson, kendall, spearman].')
    filtering_method(input_filepath, output_filepath, n_features, target_col)
    