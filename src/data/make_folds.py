# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from os import environ, mkdir
from os.path import join
from dotenv import find_dotenv, load_dotenv

generate_geo_groups_folds = __import__('1_generate_gg_folds')
generate_changing_neighbors_folds = __import__('2_generate_cn_folds')


def create_folder(path, folder_name):
    logger = logging.getLogger(__name__)
    path = join(path, folder_name)
    try:
        mkdir(path)
    except FileExistsError:
        logger.info('Folder already exist.')
    return path

def create_type_fold_folder(output_filepath, type_folds, n_neighbors):
    if type_folds == 'R':
        output_filepath = create_folder(output_filepath, 'Regions')
    elif type_folds == 'S':
        output_filepath = create_folder(output_filepath, 'States')
    elif type_folds == 'D':
        output_filepath = create_folder(output_filepath, 'Districts')
    elif type_folds == 'SD':
        output_filepath = create_folder(output_filepath, 'Sub-Districs')
    elif type_folds == 'CN':
        output_filepath = create_folder(output_filepath, 'Changing_Neighborhood')
        output_filepath = create_folder(output_filepath, n_neighbors)
        


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    # Project path
    project_dir = str(Path(__file__).resolve().parents[2])
    # Find data.env automatically by walking up directories until it's found
    dotenv_path = find_dotenv(filename='mk_folds_parameters.env')
    # Load up the entries as environment variables
    load_dotenv(dotenv_path)
    # Set paths
    input_filepath = environ.get('INPUT_DATASET')
    queen_matrix_filepath = environ.get('QUEEN_MATRIX')
    meshblock_filepath = environ.get('MESHBLOCK')
    output_filepath = environ.get('OUTPUT_PATH')
    # Set type folds configs
    type_folds = environ.get('TYPE_FOLDS')
    n_neighbors = int(environ.get('C_N_NEIGHBORS'))
    center_candidate = environ.get('CENTER_CANDIDATE')
    filter_train = environ.get('FILTER_TRAIN') 
    # Input dataset details
    region = environ.get('REGION_NAME')
    aggr = environ.get('AGGR_LEVEL')
    
    tse_year = 'E' + environ.get('ELECTION_YEAR')
    tse_office = environ.get('POLITICAL_OFFICE')
    tse_turn = 't' + str(environ.get('ELECTION_TURN'))
    tse_per = 'PER' + environ.get('PER')
    tse_candidates = environ.get('CANDIDATES').split(',')
    
    ibge_year = 'C' + environ.get('CENSUS_YEAR')
    idhm_year = 'I' + environ.get('IDHM_YEAR')
    
    
    dataset_fold_name = region +'_'+ aggr +'_'+ tse_year + tse_turn + tse_office + tse_per +'_'+ ibge_year +'_'+ idhm_year
    logger.info('Creating dataset folder.')
    output_filepath = create_folder(output_filepath, dataset_fold_name)
    if type_folds == 'CN':
        generate_changing_neighbors_folds.run(input_filepath,
                                              meshblock_filepath,
                                              output_filepath,
                                              type_folds,
                                              queen_matrix_filepath,
                                              n_neighbors,
                                              center_candidate,
                                              filter_train)
    else:
        generate_geo_groups_folds.run(input_filepath,
                                      meshblock_filepath,
                                      output_filepath,
                                      type_folds,
                                      queen_matrix_filepath)
    

