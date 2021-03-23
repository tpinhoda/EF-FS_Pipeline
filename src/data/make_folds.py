# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from os import environ
from os.path import join
from dotenv import find_dotenv, load_dotenv

from src.utils import utils
from src.data import generate_gg_folds as generate_geo_groups_folds 
from src.data import generate_cn_folds as generate_changing_neighbors_folds

#generate_geo_groups_folds = __import__('1_generate_gg_folds')
#generate_changing_neighbors_folds = __import__('2_generate_cn_folds')


def run(run_make_folds):
    logger_name = 'Spatial Folds'
    logger = logging.getLogger(logger_name)
    # Project path
    project_dir = str(Path(__file__).resolve().parents[2])
    # Find data.env automatically by walking up directories until it's found
    dotenv_path = find_dotenv(filename=join('parameters','make_folds.env'))
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
    if run_make_folds == 'True':
        logger.info('Creating dataset folder.')
        output_filepath = utils.create_folder(output_filepath, dataset_fold_name, logger_name)
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
    else:
        logger.warning('Not creating spatial folds.')
    return(dataset_fold_name)

