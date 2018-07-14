
import os
import shutil
import random
import numpy as np 
from tqdm import tqdm
from collections import OrderedDict

from config import DIR, TARGET_DIR, OUTPUT_DIR, NUM_TOP_i_IMAGES, NUM_TOP_p_Predictions

import daiquiri as dqr 
import logging 

dqr.setup(level=logging.DEBUG)
logger = dqr.getLogger()

def maybe_sample_images(dataset_name: str, sample_num: int=5):
    """Sample some images if the given directory is empty

    Args:
        dataset_name:   dataset_name
        sample_num:     number of images that randomly sampled if there is no images
                        or the directory does not exist at all.
    """
    target_dir = os.path.join(TARGET_DIR, dataset_name)
    # Create directory if not exist
    if not os.path.exists(target_dir):
        logger.warning('Target directory does not exist. Creating directory {}'.format(target_dir))
        os.makedirs(target_dir)
    
    
    listOfFiles = os.listdir(target_dir)
    logger.debug(listOfFiles)
    # Randomly sample images from dataset
    if len(listOfFiles) == 0:
        dataset_name = os.path.basename(target_dir)
        source_dir = os.path.join(DIR, dataset_name)
        rand_files = sorted([(random.random(), filename) for filename in os.listdir(source_dir)])[:sample_num]
        logger.info('Start randomly sampling images from {}'.format(source_dir))
        for (_, filename) in rand_files:
            shutil.copy(os.path.join(source_dir, filename),
                        os.path.join(target_dir, filename))
        logger.info('Finish sampling {} images'.format(sample_num))

def calculate_COS_similarity(dataset_instance_distrib: list, target_instance_distrib: list):
    """Calculate Cosine similarity between two distributions
    Args:
        dataset_instance_distrib: [('cat',0.4), ('dog',0.2),...] (1000 classes)
        target_instance_distrib:  [('cat',0.5), ('dog',0.3),...] (1000 classes)
    Return:
        Cosine similarity
    """
    np_dataset_instance = np.array([score for (_, score) in dataset_instance_distrib])
    np_target_instance  = np.array([score for (_, score) in target_instance_distrib])

    return np.dot(np_dataset_instance, np_target_instance)

def calculate_KL_similarity(dataset_instance_distrib: list, target_instance_distrib: list):
    """ Calculate Kullback-Leibler Divergence (symmetrised)
        
            D_KL(P||Q) + D_KL(Q||P)

        LAPLACE = 1e-5
        P: dataset_instance_distrib 
        Q: target_instance_distrib

    Args:
        dataset_instance_distrib: [0.4, 0.2,...] (1000 classes)
        target_instance_distrib:  [0.5, 0.3,...] (1000 classes)
    Return:
        KL Divergence
    """
    LAPLACE = 1e-5
    np_dataset_instance = np.array([score for (_, score) in dataset_instance_distrib])
    np_target_instance  = np.array([score for (_, score) in target_instance_distrib])
    
    KL = np.sum(np_dataset_instance*np.log10((np_dataset_instance+LAPLACE)/(np_target_instance+LAPLACE))) + \
         np.sum(np_target_instance*np.log10((np_target_instance+LAPLACE)/(np_dataset_instance+LAPLACE)))

    return - abs(KL)

def get_similarity_dict(target_dict: dict, dataset_dict: dict, similarity_algo: str):
    """ Get similarity dictionary between target and dataset.
    Args:
        target_dict:     {'n00x.jpg': [('cat', 0.3), ...], ...}
        dataset_dict:    {'n001.jpg': [('cat', 0.8), ...], ...}
        similarity_algo: 'COS' or 'KL'
    Return:
        sim_dict:       {
                            'n00x.jpg': (
                                [('cat', 0.3), ...], # 'n00x.jpg' distribution
                                {
                                    'n001.jpg': (
                                        [('cat', 0.8), ...], # 'n001.jpg' distribution
                                        similarity
                                    ),
                                    ...
                                }
                            )
                        }
    """
    sim_dict = {}
    logger.info('Start calculating similarity by {}'.format(similarity_algo))
    for (target_filename, target_distrib) in tqdm(target_dict.items()):
        target_sim_dict = {}
        for (dataset_filename, dataset_distrib) in dataset_dict.items():
            if similarity_algo == 'COS':
                similarity = calculate_COS_similarity(dataset_distrib, target_distrib)
                target_sim_dict[dataset_filename] = (dataset_distrib, similarity)
            elif similarity_algo == 'KL':
                similarity = calculate_KL_similarity(dataset_distrib, target_distrib)
                target_sim_dict[dataset_filename] = (dataset_distrib, similarity)
        sim_dict[target_filename] = (target_distrib, target_sim_dict)
    logger.info('Finish calculating similarity by {}'.format(similarity_algo))
    return sim_dict

def get_sorted_similarity_dict(sim_dict: dict):
    """ Get sorted similarity dictionary. Sorted by similarities.
    Args:
        sim_dict:          {
                                'n00x.jpg': (
                                    [('cat', 0.3), ...], # 'n00x.jpg' distribution
                                    {
                                        'n001.jpg': (
                                            [('cat', 0.8), ...], # 'n001.jpg' distribution
                                            similarity
                                        ),
                                        ...
                                    }
                                )
                            }  

    Return:
        sorted_sim_dict:   {
                                'n00x.jpg': (
                                    [('cat', 0.3), ...], # 'n00x.jpg' distribution
        Becomes OrderedDict  -->    {
                                        'n001.jpg': (
                                            [('cat', 0.8), ...], # 'n001.jpg' distribution
                Sort by this -->            similarity
                                        ),
                                        ...
                                    }
                                )
                            }  
    """
    sorted_sim_dict = {}

    logger.info('Start to sort similarity dict')
    for (target_filename, (target_distrib, target_sim_dict)) in tqdm(sim_dict.items()):
        sorted_target_sim_dict = OrderedDict(sorted(
            target_sim_dict.items(), key=lambda item: item[1][1], reverse=True))
        sorted_sim_dict[target_filename] = (target_distrib, sorted_target_sim_dict)
    logger.info('Finish sorting similarity dict')

    return sorted_sim_dict

def similar_copy_files(sorted_sim_dict: dict, dataset_name: str, type_: str):
    """ Copy files from dataset to output directory
    source_dir:     ./data/dataset_name
    dest_dir:       ./output/dataset_name/similar/COS or
                    ./output/dataset_name/similar/KL
    Args:
        sorted_sim_dict:   {
                                'n00x.jpg': (
                                    [('cat', 0.3), ...], # 'n00x.jpg' distribution
        Becomes OrderedDict  -->    {
                                        'n001.jpg': (
                                            [('cat', 0.8), ...], # 'n001.jpg' distribution
                Sort by this -->            similarity
                                        ),
                                        ...
                                    }
                                )
                            }
        dataset_name:       fall11_urls_top30
        type_:              'COS' or 'KL'
    """
    source_dir = os.path.join(DIR, dataset_name)
    dest_dir = os.path.join(OUTPUT_DIR, dataset_name, 'similar', type_)

    # Clean the dest_dir
    if os.path.exists(dest_dir):
        logger.info('Cleaning destination directory {}.'.format(dest_dir))
        shutil.rmtree(dest_dir, ignore_errors=True)
    os.makedirs(dest_dir)
    logger.info('Created folder {}.'.format(dest_dir))

    logger.info('Start copying images.')
    for (target_filename, (_, sorted_target_sim_dict)) in tqdm(sorted_sim_dict.items()):
        # Make target_file folder
        tar_dest_dir = os.path.join(dest_dir, target_filename)
        if not os.path.exists(tar_dest_dir):
            os.makedirs(tar_dest_dir)
        
        # Copy files
        for i, (dataset_filename, _) in enumerate(sorted_target_sim_dict.items()):
            if i >= NUM_TOP_i_IMAGES:
                break
            shutil.copy(
                os.path.join(source_dir, dataset_filename),
                os.path.join(tar_dest_dir, str(i) + '_' + dataset_filename))
    logger.info('Finish copying images')

def similar_write_files(sorted_sim_dict: dict, dataset_name: str, type_: str):
    """ Write human-readable dictionary to output directory
    dest_dir:       ./output/dataset_name/similar/COS or
                    ./output/dataset_name/similar/KL
    Args:
        sorted_sim_dict:   {
                                'n00x.jpg': (
                                    [('cat', 0.3), ...], # 'n00x.jpg' distribution
        Becomes OrderedDict  -->    {
                                        'n001.jpg': (
                                            [('cat', 0.8), ...], # 'n001.jpg' distribution
                Sort by this -->            similarity
                                        ),
                                        ...
                                    }
                                )
                            }
        dataset_name:       fall11_urls_top30
        type_:              'COS' or 'KL'          
    """
    dest_dir = os.path.join(OUTPUT_DIR, dataset_name, 'similar', type_)

    def find_top_pairs(distrib):
        return sorted(distrib, key=lambda x: x[1], reverse=True)[:NUM_TOP_p_Predictions]

    logger.info('Start to write dictionary')

    for (target_filename, (target_distrib, sorted_target_sim_dict)) in tqdm(sorted_sim_dict.items()):
        # Make target_file folder
        tar_dest_dir = os.path.join(dest_dir, target_filename)
        if not os.path.exists(tar_dest_dir):
            os.makedirs(tar_dest_dir)
        tar_dest_path = os.path.join(tar_dest_dir, 'top_' + str(NUM_TOP_i_IMAGES) + '_distributions.txt')

        # Write distribution file 
        with open(tar_dest_path, 'w') as f:
            f.write('{0} Target {1}: {0}\n'.format('='*5, target_filename))

            # Write target NUM_TOP_p_Predictions
            for i, (cate, score) in enumerate(find_top_pairs(target_distrib)):
                f.write('   {0}\t{1:.5f}:   {2}\n'.format(i+1, score, cate))
            f.write('{}\n'.format('='*30))

            # Write each instance in dataset
            f.write('{0} Images {0} {1}_Similarity {0}\n'.format('-'*4, type_))
            for i, (dataset_name, (dataset_distrib, similarity)) in enumerate(sorted_target_sim_dict.items()):
                if i >= NUM_TOP_i_IMAGES:
                    break
                f.write('   \t{}\n'.format('*'*30))
                f.write('{0}\t{1}\t{2:.5f}\n'.format(i+1, dataset_name, similarity))
                for j, (cate, score) in enumerate(find_top_pairs(dataset_distrib)):
                    f.write('\t{0}.)\t{1:.5f}: {2}\n'.format(j+1, score, cate))
    
    logger.info('Finish writing dictionary')

if __name__ == '__main__':
    maybe_sample_images('fall11_urls_top30')

    # Get dataset and target dictionaries
    from classify_images import run_predictions, get_predictions_dict
    dataset_dict = get_predictions_dict(os.path.join(DIR, 'fall11_urls_top30'))
    target_dict = get_predictions_dict(os.path.join(TARGET_DIR, 'fall11_urls_top30'))

    sim_dict = get_similarity_dict(target_dict, dataset_dict, 'COS')
    sorted_sim_dict = get_sorted_similarity_dict(sim_dict)

    similar_copy_files(sorted_sim_dict, 'fall11_urls_top30', 'COS')
    similar_write_files(sorted_sim_dict, 'fall11_urls_top30', 'COS')

    # sim_dict = get_similarity_dict(target_dict, dataset_dict, 'KL')
    # sorted_sim_dict = get_sorted_similarity_dict(sim_dict)

    # similar_copy_files(sorted_sim_dict, 'fall11_urls_top30', 'KL')
    # similar_write_files(sorted_sim_dict, 'fall11_urls_top30', 'KL')