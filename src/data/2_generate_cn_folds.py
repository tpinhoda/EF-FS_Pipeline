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

def neighbors_to_remove(indexes, matrix, data):
    matrix.index = matrix.index.astype(str)
    area_matrix = matrix.loc[indexes]
    neighbors = area_matrix.sum(axis=0) > 0
    neighbors = neighbors[neighbors].index.astype(str)
    neighbors = [n for n in neighbors if n in data.index]
    to_remove = [n for n in neighbors if n not in indexes]
    return to_remove


def generate_bipartite_matrix(data, adj_m_queen, center_candidate, n_neighbors):
    if center_candidate == 'HADDAD':
        cities_bolsonaro = data[data['ELECTION_who_won'] == 'BOLSONARO'].GEO_Cod_Municipio.astype('str').values
        cities_haddad = data[data['ELECTION_who_won'] == 'HADDAD'].GEO_Cod_Municipio.values
        adj_m_bipartite = adj_m_queen.loc[cities_haddad, cities_bolsonaro]
    else:
        cities_haddad = data[data['ELECTION_who_won'] == 'HADDAD'].GEO_Cod_Municipio.astype('str').values
        cities_bolsonaro = data[data['ELECTION_who_won'] == 'BOLSONARO'].GEO_Cod_Municipio.values
        adj_m_bipartite = adj_m_queen.loc[cities_bolsonaro, cities_haddad]
    
    change_cities = adj_m_bipartite.sum(axis=1) > n_neighbors    
    change_cities = [int(id) for id in change_cities.index if change_cities.loc[id] == True]
    adj_m_bipartite = adj_m_queen.loc[change_cities,].copy()
    adj_m_bipartite = adj_m_bipartite[adj_m_bipartite.columns[adj_m_bipartite.sum()>0]]
    return adj_m_bipartite


def filter_data(data, adj_m_bipartite, output_filepath):
    center_cities = adj_m_bipartite.index.astype('str').values.tolist()
    neighbor_cities = adj_m_bipartite.columns.values.tolist()
    filtered_cities = center_cities + neighbor_cities
    data_filtered = data.loc[filtered_cities,].copy()
    data_filtered.to_csv(join(output_filepath, 'filtered_data.csv'))
    return data_filtered
        

def make_folds_by_changing_neighborhood(input_filepath, queen_matrix_filepath, meshblock_filepath, output_filepath, type_folds, center_candidate, n_neighbors, filter_train):
    logger = logging.getLogger(__name__)
    logger.info('Generating Changing Neighborhood folds')
    output_filepath = utils.create_folder(output_filepath, 'Changing_Neighborhood')
    output_filepath = utils.create_folder(output_filepath, center_candidate+'_N'+str(n_neighbors)+'_FT_'+filter_train)
    
    data = pd.read_csv(input_filepath)
    adj_m_queen = pd.read_csv(queen_matrix_filepath)
    meshblock = gpd.read_file(meshblock_filepath)
    
    
    if data.columns[0] == 'GEO_Cod_Municipio':
        index_name = 'CD_GEOCMU'
        data[index_name] = data.GEO_Cod_Municipio.astype('str')
    elif data.columns[0] == 'GEO_Cod_ap':
        index_name = 'Cod_ap'
        data[index_name] = data.GEO_Cod_ap.astype('str')
    
    data.set_index(index_name, inplace=True)
    adj_m_queen.set_index(index_name, inplace=True)
    
    meshblock.set_index(index_name, inplace=True)
    meshblock = meshblock.merge(data['ELECTION_who_won'], on=index_name, how='left')
    
    adj_m_bipartite = generate_bipartite_matrix(data, adj_m_queen, center_candidate, n_neighbors)

    if filter_train == 'True':
        data = filter_data(data, adj_m_bipartite,  output_filepath)   
    fold_output_filepath = utils.create_folder(output_filepath, 'folds')    
    for center_city_idx, row in tqdm(adj_m_bipartite.iterrows(), total=len(adj_m_bipartite)):
        test_idx =  row >= 1
        test_idx = test_idx[test_idx].index.values.tolist()
        test_idx.append(str(center_city_idx))
        test_data = data.loc[test_idx,]
        #Generete column to know the center
        test_data['center_neighbor'] = ['neighbor'] * len(test_data)
        test_data.loc[str(center_city_idx), 'center_neighbor'] = 'center'
        
        fold_path = utils.create_folder(fold_output_filepath, str(center_city_idx).lower())
        train_data = data.copy()
            
        train_data.drop(test_data.index, inplace=True)
        buffer = neighbors_to_remove(test_data.index, adj_m_queen, data)
        train_data.drop(buffer, inplace=True)
            
        test_data.to_csv(join(fold_path, 'test.csv'))
        train_data.to_csv(join(fold_path, 'train.csv'))
    return output_filepath
    
 
def plot_cn_groups(input_filepath, meshblock_filepath, queen_matrix_filepath, output_filepath, center_candidate, n_neighbors):
    data = pd.read_csv(input_filepath)
    adj_m_queen = pd.read_csv(queen_matrix_filepath)
    meshblocks = gpd.read_file(meshblock_filepath)
    
    if data.columns[0] == 'GEO_Cod_Municipio':
        index_name = 'CD_GEOCMU'
        data[index_name] = data.GEO_Cod_Municipio.astype('str')
    elif data.columns[0] == 'GEO_Cod_ap':
        index_name = 'Cod_ap'
        data[index_name] = data.GEO_Cod_ap.astype('str')
    
    adj_m_queen.set_index(index_name, inplace=True)
    
    meshblocks = meshblocks.merge(data[['ELECTION_who_won', index_name]], on=index_name, how='left')
    meshblocks.set_index(index_name, inplace=True)
    
    adj_m_bipartite = generate_bipartite_matrix(data, adj_m_queen, center_candidate, n_neighbors)

    filtered_cities = adj_m_bipartite.index.astype(str).values.tolist() + adj_m_bipartite.columns.values.tolist()
    center_cities = adj_m_bipartite.index.astype(str).values.tolist()
    meshblocks.loc[center_cities, 'ELECTION_who_won'] = 'Center City'
    meshblocks = meshblocks.loc[filtered_cities]
    _, ax = plt.subplots(1, 1)
    meshblocks.plot(column='ELECTION_who_won', cmap='Paired', legend=True, ax=ax)
    plt.savefig(join(output_filepath, 'CN_folds.pdf'))
    
    
def run(input_filepath, meshblock_filepath, output_filepath, type_folds, queen_matrix_filepath, n_neighbors, center_candidate, filter_train):
    output_filepath = make_folds_by_changing_neighborhood(input_filepath, queen_matrix_filepath, meshblock_filepath, output_filepath, type_folds, center_candidate, n_neighbors, filter_train)
    plot_cn_groups(input_filepath, meshblock_filepath, queen_matrix_filepath, output_filepath, center_candidate, n_neighbors)