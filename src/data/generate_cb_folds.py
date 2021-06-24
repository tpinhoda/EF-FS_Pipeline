# -*- coding: utf-8 -*-
import logging
import warnings
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import time
from os.path import join
from statistics import mean
from tqdm import tqdm
from geopy.distance import distance
from scipy.spatial.distance import minkowski
from src.utils import utils
from operator import itemgetter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
warnings.filterwarnings('ignore')

def neighbors_to_remove(spatial_attr, area, indexes, matrix, data):
    indexes = [int(i) for i in indexes]
    area_matrix = matrix.loc[indexes]
    neighbors = area_matrix.sum(axis=0) > 0
    neighbors = neighbors[neighbors].index.astype('int64')
    neighbors = [n for n in neighbors if n not in indexes]
 
    return neighbors


def calculate_longest_path(area_indexes, matrix):
    neighbors = neighbors_to_remove(0,0, area_indexes, matrix,0)
    size_tree = 0
    while len(neighbors) > 0:
        size_tree += 1
        neighbors = neighbors_to_remove(0,0,area_indexes, matrix,0)
        area_indexes = area_indexes + neighbors
    return size_tree

def distance_gamma(x, y, geo_dist):
    return ((x - y)**2) * geo_dist


def haversine_np(lon1, lat1, lon2, lat2):

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

def calculate_diff_test(t, context_neighbors, attribute, based_dist):
    #context_neighbors['geo_dist'] = 1 / context_neighbors.apply(lambda row: distance((t['GEO_x'], t['GEO_y']), (row['GEO_x'], row['GEO_y'])).km, axis=1)
    if based_dist:
        context_neighbors['geo_dist'] = haversine_np(np.array(t['GEO_x']), np.array(t['GEO_y']), context_neighbors['GEO_x'].values, context_neighbors['GEO_y'].values)
    else:
        context_neighbors['geo_dist'] = np.ones(len(context_neighbors)) 
    context_neighbors['gamma_dist'] = (t[attribute] - context_neighbors[attribute])**2
    context_neighbors['gamma_dist'] = context_neighbors['gamma_dist'] * context_neighbors['geo_dist'] 
    #context_neighbors.apply(lambda row: distance_gamma(t[attribute], row[attribute], row['geo_dist']), axis=1) 
    return pd.Series({'geo_dist': context_neighbors['geo_dist'].sum(), 'gamma_dist': context_neighbors['gamma_dist'].sum()})
    

def calculate_diff_test_mean(mean, context_neighbors, attribute, based_dist):
    #context_neighbors['geo_dist'] = 1 / context_neighbors.apply(lambda row: distance((t['GEO_x'], t['GEO_y']), (row['GEO_x'], row['GEO_y'])).km, axis=1)
    if based_dist:
        context_neighbors['geo_dist'] = haversine_np(np.array(t['GEO_x']), np.array(t['GEO_y']), context_neighbors['GEO_x'].values, context_neighbors['GEO_y'].values)
    else:
        context_neighbors['geo_dist'] = np.ones(len(context_neighbors))
    print('====')
    print(mean)
    print(context_neighbors[attribute])
    context_neighbors['gamma_dist'] = (context_neighbors[attribute] - mean)**2
    print(context_neighbors['gamma_dist'])
    print(context_neighbors['geo_dist'])
    
    context_neighbors['gamma_dist'] = context_neighbors['gamma_dist'] * context_neighbors['geo_dist'] 
    print(context_neighbors['gamma_dist'])
    exit()
    #context_neighbors.apply(lambda row: distance_gamma(t[attribute], row[attribute], row['geo_dist']), axis=1) 
    return context_neighbors['gamma_dist'].sum()/context_neighbors['geo_dist'].sum()  * 0.5


def calculate_gamma(data, context_attr, test, neighbors, attribute, based_dist):
    context_gamma = {}
    test_data = data.loc[test]
    neighbors = [n for n in neighbors if n in data.index] 
    neighbors_data = data.loc[neighbors]
    #print(test_data[attribute])
    for context, context_neighbors in neighbors_data.groupby(by=context_attr, as_index=False):
        #print(context)
        diff_test = test_data.apply(lambda row: calculate_diff_test(row, context_neighbors, attribute, based_dist), axis=1)
        sum_diff = diff_test['gamma_dist'].sum()
        sum_dist = diff_test['geo_dist'].sum()
        
        gamma = (sum_diff/(2*sum_dist))
        #print(gamma)
        #gamma = calculate_diff_test_mean(test_data[attribute].mean(), context_neighbors, attribute, based_dist)
        context_gamma[context] = {'gamma': gamma, 'neighbors': context_neighbors.index.values.tolist()}
    return context_gamma

def calculate_local_variance(data_1, data_2, attribute, n_time):
    list_var = []
    for _ in range(n_time):
        if len(data_1) > len(data_2):
            data_1 = data_1.sample(len(data_2))
        else:
            data_2 = data_2.sample(len(data_1))
        
        data = pd.concat([data_1, data_2])
        list_var.append(data[attribute].var())
    return mean(list_var)
        
    
    
         
    
def choose_near_contexts(test, train, vizinhos, geo_context, fold_name, logger):       
        geo_dist = dict()
        geo_w = 500
        test['GEO_x'] = (test['GEO_x']/100) * geo_w
        test['GEO_y'] = (test['GEO_y']/100) * geo_w
        geo_group = utils.get_geo_attribute(geo_context, logger)
        
        cols_valid = [c for c in test.columns if 'GEO' not in c]
        cols_valid = [c for c in cols_valid if 'LISA' not in c]
        cols_valid = [c for c in cols_valid if 'BOLSONARO' not in c]
        cols_valid = [c for c in cols_valid if 'HADDAD' not in c]
        cols_valid = [c for c in cols_valid if 'who_won' not in c]
        cols_valid = ['GEO_x', 'GEO_y']


        data_group = train.groupby(by=geo_group)
        for key, context in train.groupby(by=geo_group):
            context['GEO_x'] = (context['GEO_x'] / 100) * geo_w
            context['GEO_y'] = (context['GEO_y'] / 100) * geo_w
            geo_dist[key] = minkowski(test[cols_valid].mean(), context[cols_valid].mean(), 2)

        context_chosen = sorted(geo_dist.items(), key=itemgetter(1))
     
        list_groups = []
        minority_perc = 0.25
        percs_won = 0
        percs_lost = 0
        len_train = 2
        list_context = []
        for context, _ in context_chosen:
            if  (percs_won <= minority_perc or percs_lost <= minority_perc) or vizinhos[fold_name] * len_train > len(list_context): 
                list_context.append(context)
                list_groups.append(data_group.get_group(context))
                train = pd.concat(list_groups, axis=0)
                train['won_lost'] = np.where(train['ELECTION_JAIR BOLSONARO(%)'] > 50, 'won', 'lost')
                percs = train['won_lost'].value_counts(normalize=True)
              
                try:
                    percs_won = percs['won']
                except KeyError:
                    percs_won = 0
                try:
                    percs_lost = percs['lost']
                except KeyError:
                    percs_lost = 0
        print(' {}'.format(list_context))
        print(percs)      
        return train
    
    
def calculate_buffer(geo_group, fold_name, test_data, adj_m_queen, data, vizinhos, sill, target_attr, kappa=20, relax_factor=0, based_dist = 1, train = 0):
    gamma = 0
    #relax_factor = 0.003
    neighbors = []
    buffer = []
    buffer_test = []
    count_n = 0
    at_least_one = 1
    max_context = 4
    list_context = []
    size_tree = calculate_longest_path(test_data.index.tolist(), adj_m_queen)

    while at_least_one:
        at_least_one = 0
        start = time.process_time()
        neighbors = neighbors_to_remove(geo_group, fold_name, test_data.index.values.tolist() + buffer , adj_m_queen, data)
        #print(time.process_time() - start)
        #print('======================')        
        start = time.process_time()
        context_gamma = calculate_gamma(data, geo_group, test_data.index, neighbors, target_attr, based_dist)
        #print(context_gamma)
        #print(time.process_time() - start)
        #exit()
        for context in context_gamma:
            gamma = context_gamma[context]['gamma']
            exponent = np.log(1*size_tree - count_n) / np.log(1*size_tree)
            sill_value = sill[context]*exponent
            #print(sill_value)    
            if gamma  <= sill_value and len(set(list_context + [context])) <= vizinhos[fold_name] * kappa:  
                list_context.append(context)
                at_least_one = 1
                buffer = buffer + context_gamma[context]['neighbors']
           # elif len(buffer)  == 0 and train:
           #     sill[context] = sill[context] + abs(sill[context] - context_gamma[context]['gamma'])
           #     list_context.append(context)
           #     at_least_one = 1
           #     count_n += 1
           #     buffer = buffer + context_gamma[context]['neighbors']
        count_n += 1        
                
        
    return buffer
    
def make_folds_by_context_group(input_filepath, output_filepath, queen_matrix_filepath, geo_context, filter_data, filter_attr, filter_value, meshblock_filepath):
    
    vizinhos = {
            'Acre': 2,
            'Amazonas': 5,
            'Roraima': 2,
            'Rondônia': 4,
            'Pará': 6,
            'Amapá': 1,
            'Tocantins':6,
            'Mato Grosso':6,
            'Goiás':6,
            'Mato Grosso do Sul':5,
            'Distrito Federal':1,
            'Paraná':3,
            'Santa Catarina':2,
            'Rio Grande do Sul':1,
            'São Paulo':5,
            'Rio de Janeiro':3,
            'EspÃ\xadrito Santo':3,
            'Minas Gerais':6,
            'Bahia':9,
            'Sergipe':2,
            'Alagoas':3,
            'Pernambuco':5,
            'Paraiba':5,
            'Rio Grande do Norte':5,
            'Ceará':4,
            'Piauí':4,
            'Maranhão':4
    }
        
    
    logger_name = 'Spatial Folds'
    logger = logging.getLogger(logger_name)
    adj_m_queen = pd.read_csv(queen_matrix_filepath)
    adj_m_queen.set_index(adj_m_queen.columns[0], inplace=True)
    data = pd.read_csv(input_filepath)
    data.set_index(data.columns[0], inplace=True)
    if filter_data == 'True':
        data = data[data[filter_attr].astype('str') == filter_value]
        data.to_csv(join(output_filepath,'filtered_data.csv'))
        
    geo_group = utils.get_geo_attribute(geo_context, logger_name)
    
    census_col = [c for c in data.columns if 'CENSUS' in c]
    pca = PCA(n_components=1)
    pca.fit(data[census_col])
    comp = pca.transform(data[census_col])
    data['PCA'] = comp
    data['ELECTION_JAIR BOLSONARO(%)'] = data['ELECTION_JAIR BOLSONARO(%)']/100
    #tsne = TSNE(n_components=1, verbose=1, perplexity=40, n_iter=300)
    #data['TSNE'] = tsne.fit_transform(data[census_col])
    
    logger.info('Generating spatial folds by: {}'.format(geo_group))
    output_filepath = utils.create_folder(output_filepath, 'Geographical Context', logger_name)
    name_fold = geo_group.split('_')[-1]
    output_filepath = utils.create_folder(output_filepath, name_fold, logger_name)
    fold_output_filepath = utils.create_folder(output_filepath, 'folds', logger_name)
    for key, test_data in tqdm(data.groupby(by=geo_group), desc='Creating spatial folds', leave=False):
        print(' ' + key)
        if len(test_data) >= 1:
            fold_path = utils.create_folder(fold_output_filepath, str(key).lower(), logger_name, show_msg=False)
            train_data = data.copy()
            train_data.drop(test_data.index, inplace=True)
            sill_test = {}
            sill_train = {}
            local_sill = {}
            #sill_overall = (data['PCA']).var()  
            for context, context_data in data.groupby(geo_group, as_index=False):
                concat_data = pd.concat([test_data, context_data])
                sill_test[context] =  concat_data['ELECTION_JAIR BOLSONARO(%)'].var()
                sill_train[context] = concat_data['PCA'].var()
                #sill_test[context] = (test_data['ELECTION_JAIR BOLSONARO(%)'].var() + context_data['ELECTION_JAIR BOLSONARO(%)'].var())/2 
                #sill_train[context] = (test_data['PCA'].var() + context_data['PCA'].var())/2
                
                #sill_test[context] = calculate_local_variance(context_data, test_data, 'ELECTION_JAIR BOLSONARO(%)', 100)
                #sill_train[context] = calculate_local_variance(context_data, test_data, 'PCA', 100)
            #print(sill_test)  
            #print(sill_test)
#            print('================')
#            print(local_sill)
#            exit()
            #print('====================')
           # print(sill_train)
            max_var_train =  max(sill_train, key=sill_train.get)
            for c in sill_train:
                sill_train[c] = sill_train[max_var_train]
            
            #print(sill_train)
             
            buffer_training = calculate_buffer(geo_group, key, test_data, adj_m_queen, data, vizinhos, sill_train, 'PCA', kappa=20, relax_factor=1, based_dist=0, train = 1)        
            buffer_training = list(set(buffer_training))
            
            if buffer_training:
                train_data = train_data.loc[buffer_training]
        
            fold_data = pd.concat([train_data, test_data])
            print('Buffer Traininig: {}'.format(len(buffer_training)))
          
            buffer_test = calculate_buffer(geo_group, key, test_data, adj_m_queen, fold_data, vizinhos, sill_test, 'ELECTION_JAIR BOLSONARO(%)', relax_factor=0, based_dist=0)
            buffer_test = list(set(buffer_test))
            train_data.drop(buffer_test, inplace=True)
            
            print('Buffer test: {}'.format(len(buffer_test)))         
            
            


            plot_folds(train_data, test_data, data.loc[buffer_test], data, meshblock_filepath, fold_path)
            
            #train_data = choose_near_contexts(test_data, train_data, vizinhos, geo_context, key, logger_name)
            test_data.to_csv(join(fold_path, 'test.csv'))
            train_data.to_csv(join(fold_path, 'train.csv'))
    return output_filepath


def plot_folds(train, test, buffer_test, data, meshblock_filepath, output_filepath):
    logger_name = 'Spatial Folds'
    logger = logging.getLogger(logger_name)
    
    meshblock = gpd.read_file(meshblock_filepath)
    train['type_data'] = ['train'] * len(train)
    test['type_data'] = ['test'] * len(test)
    
    buffer_test['type_data'] = ['buffer_test'] * len(buffer_test)
    list_index = pd.concat([train, test, buffer_test]).index
    buffer_training_ind = [ i for i in data.index if i not in list_index]
    buffer_training = data.loc[buffer_training_ind]
    buffer_training['type_data'] = ['buffer_training'] * len(buffer_training)
    
    data = pd.concat([train, test, buffer_test, buffer_training])
    
    
    index_name = test.index.name
    data.reset_index(inplace=True)
    
    if index_name == 'GEO_Cod_Municipio':
        mesh_index = 'CD_GEOCMU'
    elif index_name == 'GEO_Cod_ap':
        mesh_index = 'Cod_ap'
        
    meshblock.rename(columns={mesh_index: index_name}, inplace=True)  
    
    meshblock[index_name] = meshblock[index_name].astype('int64')
    data[index_name] = data[index_name].astype('int64')
    
    meshblock = meshblock.merge(data[[index_name,'type_data']], on=index_name, how='left')
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
    plt.savefig(join(output_filepath, 'folds.png'), dpi=1000)

def map_color(row):
    if row['type_data'] == 'train':
        return '#f9cb9cff'
    elif row['type_data'] == 'test':
        return '#cfe2f3ff'
    elif row['type_data'] == 'buffer_test':
        return '#e69138ff'
    elif row['type_data'] == 'buffer_training':
        return '#ef233c'
    else:
        return '#ffffffff'    

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
    meshblock.plot(column=geo_attr, categorical=True, cmap='Set1', linewidth=.1, edgecolor='0.2',
                   legend=True, legend_kwds={'bbox_to_anchor': (.3, 1.05), 'fontsize': 16, 'frameon': False}, ax=ax)
    plt.savefig(join(output_filepath, '{}_folds.pdf'.format(name_fold)))
    

def run(input_filepath, meshblock_filepath, output_filepath, geo_context, queen_matrix_filepath, filter_data, filter_attr, filter_value):
    # Log text to show on screen
    output_filepath = make_folds_by_context_group(input_filepath, output_filepath, queen_matrix_filepath, geo_context, filter_data, filter_attr, filter_value, meshblock_filepath)
    plot_geo_groups(input_filepath, meshblock_filepath, output_filepath, geo_context, filter_data, filter_attr, filter_value)
