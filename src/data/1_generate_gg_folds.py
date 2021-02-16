# -*- coding: utf-8 -*-
import coloredlogs,  logging
coloredlogs.install()
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
import warnings
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from os.path import join
from tqdm import tqdm
from src.utils import utils
warnings.filterwarnings('ignore')

def neighbors_to_remove(spatial_attr, area, indexes, matrix, data):
    area_matrix = matrix.loc[indexes]
    neighbors = area_matrix.sum(axis=0) > 0
    neighbors = neighbors[neighbors].index.astype('int64')
    neighbors = [n for n in neighbors if n in data.columns]
    neighbors_data = data.loc[neighbors]
    to_remove = neighbors_data[neighbors_data[spatial_attr] != area].index
    return to_remove

def make_folds_by_geographic_group(input_filepath, output_filepath, queen_matrix_filepath, type_folds):
    logger = logging.getLogger(__name__)
    adj_m_queen = pd.read_csv(queen_matrix_filepath)
    adj_m_queen.set_index(adj_m_queen.columns[0], inplace=True)
    data = pd.read_csv(input_filepath)
    data.set_index(data.columns[0], inplace=True)
    
    geo_group = utils.get_geo_attribute(type_folds)
    
    logger.info('Generating spatial folds by: {}'.format(geo_group))
    name_fold = geo_group.split('_')[-1]
    output_filepath = utils.create_folder(output_filepath, name_fold)
    fold_output_filepath = utils.create_folder(output_filepath, 'folds')
    for key, test_data in tqdm(data.groupby(by=geo_group)):
        if len(test_data) >= 1:
            fold_path = utils.create_folder(fold_output_filepath, str(key).lower())
            train_data = data.copy()
            
            train_data.drop(test_data.index, inplace=True)
            buffer = neighbors_to_remove(geo_group, key, test_data.index, adj_m_queen, data)
            train_data.drop(buffer, inplace=True)
            
            test_data.to_csv(join(fold_path, 'test.csv'))
            train_data.to_csv(join(fold_path, 'train.csv'))
    return output_filepath
            

def plot_geo_groups(input_filepath, meshblock_filepath, output_filepath, type_folds):
    logger = logging.getLogger(__name__)
    meshblock = gpd.read_file(meshblock_filepath)
    data = pd.read_csv(input_filepath) 
    index_name = data.columns[0]
    if index_name == 'GEO_Cod_Municipio':
        mesh_index = 'CD_GEOCMU'
    elif index_name == 'GEO_Cod_ap':
        mesh_index = 'Cod_ap'
        
    meshblock.rename(columns={mesh_index: index_name}, inplace=True)  
    
    meshblock[index_name] = meshblock[index_name].astype('int64')
    data[index_name] = data[index_name].astype('int64')
    
    geo_attr = utils.get_geo_attribute(type_folds)
    name_fold = geo_attr.split('_')[-1]
    logger.info('Plotting spatial folds by: {}'.format(geo_attr))
    
    meshblock = meshblock.merge(data[[index_name,geo_attr]], on=index_name, how='left')
    fig, ax = plt.subplots(1, 1)
    meshblock.plot(column=geo_attr, categorical=True, cmap='tab20', linewidth=.6, edgecolor='0.2',
                   legend=False, legend_kwds={'bbox_to_anchor': (.3, 1.05), 'fontsize': 16, 'frameon': False}, ax=ax)
    plt.savefig(join(output_filepath, '{}_folds.pdf'.format(name_fold)))
    

def run(input_filepath, meshblock_filepath, output_filepath, type_folds, queen_matrix_filepath):
    # Log text to show on screen
    output_filepath = make_folds_by_geographic_group(input_filepath, output_filepath, queen_matrix_filepath, type_folds)
    plot_geo_groups(input_filepath, meshblock_filepath, output_filepath, type_folds)
