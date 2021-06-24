# -*- coding: utf-8 -*-
import logging
import warnings
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from os.path import join
from tqdm import tqdm
from src.utils import utils
warnings.filterwarnings('ignore')

def neighbors_to_remove(indexes, matrix):
    indexes = [int(i) for i in indexes]
    area_matrix = matrix.loc[indexes]
    neighbors = area_matrix.sum(axis=0) > 0
    neighbors = neighbors[neighbors].index.astype('int64')
    neighbors = [n for n in neighbors if n not in indexes]
    return neighbors


def calculate_variogram(values, matrix, lags):
    vario_lags = {}
    sill = (values/100).var()
    i = 0
    gamma = -100000
    print(sill)
    while gamma < sill:
        i += 1
        count_pairs = 0
        sum_pairs = 0
        print(i)
        for key, v in values.iteritems():
            list_nodes = [key]
            for k in range(i):
                if k < i-1:
                    neighbors = neighbors_to_remove(list_nodes, matrix)
                    list_nodes = list_nodes + neighbors
                else:
                    neighbors = neighbors_to_remove(list_nodes, matrix)
                    for n in neighbors:
                        count_pairs += 1
                        sum_pairs += (v/100 - values[n]/100)**2
        print(neighbors)
        print(count_pairs)
        gamma = sum_pairs/(count_pairs) * 0.5
        vario_lags[i] = gamma
        print(vario_lags)
    exit()                    
                    
                    
                    
    

def make_folds_by_geographic_group(input_filepath, output_filepath, queen_matrix_filepath, type_folds, filter_data, filter_attr, filter_value, meshblock_filepath):
    logger_name = 'Spatial Folds'
    logger = logging.getLogger(logger_name)
    adj_m_queen = pd.read_csv(queen_matrix_filepath)
    adj_m_queen.set_index(adj_m_queen.columns[0], inplace=True)
    data = pd.read_csv(input_filepath)
    data.set_index(data.columns[0], inplace=True)
    data['ELECTION_JAIR BOLSONARO(%)'] = data['ELECTION_JAIR BOLSONARO(%)']/100
    if filter_data == 'True':
        data = data[data[filter_attr].astype('str') == filter_value]
        data.to_csv(join(output_filepath,'filtered_data.csv'))
    
    #calculate_variogram(data['ELECTION_JAIR BOLSONARO(%)'], adj_m_queen, 4)    
    geo_group = utils.get_geo_attribute(type_folds, logger_name)
    logger.info('Generating spatial folds by: {}'.format(geo_group))
    output_filepath = utils.create_folder(output_filepath,  'Geographical Group', logger_name)
    name_fold = geo_group.split('_')[-1]
    output_filepath = utils.create_folder(output_filepath,  name_fold, logger_name)
    fold_output_filepath = utils.create_folder(output_filepath, 'folds', logger_name)
    for key, test_data in tqdm(data.groupby(by=geo_group), desc='Creating spatial folds', leave=False):
        if len(test_data) >= 1 :
            fold_path = utils.create_folder(fold_output_filepath, str(key).lower(), logger_name, show_msg=False)
            train_data = data.copy()
            
            train_data.drop(test_data.index, inplace=True)
            test_region = test_data.index.values.tolist()
            buffer = []
            for i in range(27):
                neighborhood = neighbors_to_remove(test_region, adj_m_queen)
                buffer = buffer + neighborhood
                test_region = test_region + neighborhood
            
            buffer = list(set(buffer))
            plot_folds(train_data, test_data, data.loc[buffer], meshblock_filepath, fold_path)
            train_data.drop(buffer, inplace=True)
            test_data.to_csv(join(fold_path, 'test.csv'))
            train_data.to_csv(join(fold_path, 'train.csv'))
    return output_filepath


def plot_folds(train, test, buffer, meshblock_filepath, output_filepath):
    logger_name = 'Spatial Folds'
    logger = logging.getLogger(logger_name)
    meshblock = gpd.read_file(meshblock_filepath)
    train['type_data'] = ['train'] * len(train)
    test['type_data'] = ['test'] * len(test)
    buffer['type_data'] = ['buffer'] * len(buffer)
    data = pd.concat([train, test, buffer])
    
    index_name = test.index.name
    data.reset_index(inplace=True)
    
    if index_name == 'GEO_Cod_Municipio':
        mesh_index = 'CD_GEOCMU'
    elif index_name == 'GEO_Cod_ap':
        mesh_index = 'Cod_ap'
        
    meshblock.rename(columns={mesh_index: index_name}, inplace=True)  
    
    meshblock[index_name] = meshblock[index_name].astype('int64')
    data[index_name] = data[index_name].astype('int64')
    
    meshblock = meshblock.merge(data[[index_name,'type_data', 'ELECTION_who_won']], on=index_name, how='left')
    fig, ax = plt.subplots(1, 1)
    color_list = meshblock.apply(lambda row: map_color(row), axis=1)
    
    meshblock.plot(column= 'type_data', 
                   categorical=True, 
                   color=color_list, 
                   linewidth=.05, 
                   edgecolor='0.2',
                   legend=False, 
                   legend_kwds={'bbox_to_anchor': (.3, 1.05), 
                                'fontsize': 16, 
                                'frameon': False}, 
                   ax=ax)
    plt.savefig(join(output_filepath, 'folds.png'), dpi=1500)

def map_color(row):
    if row['type_data'] == 'train':
        return '#f9cb9cff'
    elif row['type_data'] == 'test':
        return '#cfe2f3ff'
    elif row['type_data'] == 'buffer':
        return '#e69138ff'
    else:
        return '#ffffffff'
    

def map_color_won(row):
    if row['ELECTION_who_won'] == 'BOLSONARO':
        return '#b6d7a8ff'
    else:
        return '#ea9999ff'             

def plot_geo_groups(input_filepath, meshblock_filepath, output_filepath, type_folds, filter_data, filter_attr, filter_value):
    logger_name = 'Spatial Folds'
    logger = logging.getLogger(logger_name)
    meshblock = gpd.read_file(meshblock_filepath)
    data = pd.read_csv(input_filepath)
    if filter_data == 'True':
        data = data[data[filter_attr].astype('str') == filter_value]
        data.to_csv(join(output_filepath,'filtered_data.csv'))
    index_name = data.columns[0]
    if index_name == 'GEO_Cod_Municipio':
        mesh_index = 'CD_GEOCMU'
    elif index_name == 'GEO_Cod_ap':
        mesh_index = 'Cod_ap'
        
    meshblock.rename(columns={mesh_index: index_name}, inplace=True)  
    
    meshblock[index_name] = meshblock[index_name].astype('int64')
    data[index_name] = data[index_name].astype('int64')
    
    geo_attr = utils.get_geo_attribute(type_folds, logger_name)
    name_fold = geo_attr.split('_')[-1]
    logger.info('Plotting spatial folds by: {}'.format(geo_attr))
    
    meshblock = meshblock.merge(data[[index_name,geo_attr]], on=index_name, how='left')
    fig, ax = plt.subplots(1, 1)
    meshblock.plot(column=geo_attr, categorical=True, cmap='tab20', linewidth=.6, edgecolor='0.2',
                   legend=False, legend_kwds={'bbox_to_anchor': (.3, 1.05), 'fontsize': 16, 'frameon': False}, ax=ax)
    plt.savefig(join(output_filepath, '{}_folds.pdf'.format(name_fold)))
    

def run(input_filepath, meshblock_filepath, output_filepath, type_folds, queen_matrix_filepath, filter_data, filter_attr, filter_value):
    # Log text to show on screen
    output_filepath = make_folds_by_geographic_group(input_filepath, output_filepath, queen_matrix_filepath, type_folds, filter_data, filter_attr, filter_value, meshblock_filepath)
    plot_geo_groups(input_filepath, meshblock_filepath, output_filepath, type_folds, filter_data, filter_attr, filter_value)
