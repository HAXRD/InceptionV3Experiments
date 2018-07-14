import os
import sys
import shutil
from tqdm import tqdm
import numpy as np 
import tensorflow as tf 

from collections import OrderedDict
from config import *

import daiquiri as dqr 
import logging

dqr.setup(level=logging.DEBUG)
logger = dqr.getLogger()

def sort_dictionary_vals(predictions_dict):
    """Iterate through the 'predictions_dict' and sort each item."""
    # {'n001.jpg': [('dog', 0.5),...], ...}
    val_sorted_dict = {}
    for key, cate_score_pairs in predictions_dict.items():
        val_sorted_dict[key] = sorted(cate_score_pairs, key=lambda x: x[1], reverse=True)
    return val_sorted_dict

def find_confident_images(predictions_dict):
    """
        1. Call 'sort_dictionary_vals' 
        2. Sort the key-val pair into OrderedDict by the value of the first score of cate_score_pairs.
    """
    # 1.
    val_sorted_dict = sort_dictionary_vals(predictions_dict)
    # 2.
    sorted_by_confidence_dict = OrderedDict(sorted(
        val_sorted_dict.items(), key=lambda item: item[1][0][1], reverse=True))

    return sorted_by_confidence_dict

def find_ambiguous_images(predictions_dict, top_k, threshold):
    """ Distribution
        ▊
        ▊ ▊ ▊
        ▊ ▊ ▊ ▊ ▊
        ▊ ▊ ▊ ▊ ▊
        1. Call 'sort_dictionary_vals'
        2. Filter out those sum(top_k) < threshold
        3. Sort by variance.
        This is a generalization version of the method by comparing the ratio.
        By specifying top_k, e.g. top_k = 2, result would be bi-model.
    """
    # 1.
    val_sorted_dict = sort_dictionary_vals(predictions_dict)

    # 2.
    filtered_dict = {}
    for key, cate_score_pairs in val_sorted_dict.items():
        top_k_sum = np.sum(np.array([pair[1] for pair in cate_score_pairs])[:top_k])
        if top_k_sum*100. >= threshold:
            filtered_dict[key] = cate_score_pairs
    # 3.
    def comp(item):
        cate_score_pairs = item[1]
        top_k_scores = np.array([pair[1] for pair in cate_score_pairs])[:top_k]
        top_k_mean = np.mean(top_k_scores)
        top_k_variance = np.mean(np.square(top_k_scores - top_k_mean))
        return top_k_variance

    sorted_by_ambiguous_dict = OrderedDict(sorted(
        filtered_dict.items(), key=comp))
    return sorted_by_ambiguous_dict

def ambiguous_copy_files(given_dict: dict, dataset_name: str, type_: str):
    """ Copy images from dataset to output directory
    source_dir:         ./data/dataset_name/
    dest_dir:           ./output/dataset_name/ambiguous/confident or
                        ./output/dataset_name/ambiguous/ambiguous

    Args:
        given_dict:     sorted_by_confidence_dict
        dataset_name:   fall11_urls_top30
        type_:          ambiguous or confident
    """
    source_dir = os.path.join(DIR, dataset_name)
    dest_dir = os.path.join(OUTPUT_DIR, dataset_name, 'ambiguous', type_)

    # Clean the dest_dir
    if os.path.exists(dest_dir):
        logger.info('Cleaning destination directory {}.'.format(dest_dir))
        shutil.rmtree(dest_dir, ignore_errors=True)
    os.makedirs(dest_dir)
    logger.info('Created folder {}.'.format(dest_dir))

    logger.info('Start copying images.')
    for i, (filename, _) in tqdm(enumerate(given_dict.items())):
        if i >= NUM_TOP_i_IMAGES:
            break
        shutil.copyfile(
            os.path.join(source_dir, filename),
            os.path.join(dest_dir, str(i) + '_' + filename))
    logger.info('Finish copying images')
    
def ambiguous_write_dict(given_dict: dict, dataset_name: str, type_: str):
    """ Write human-readable dictionary to output directory
    dest_dir:           ./output/dataset_name/ambiguous/confident or
                        ./output/dataset_name/ambiguous/ambiguous

    Args:
        given_dict:     sorted_by_confidence_dict
        dataset_name:   fall11_urls_top30
        type_:          ambiguous or confident
    """
    dest_dir = os.path.join(OUTPUT_DIR, dataset_name, 'ambiguous', type_)
    dict_filename = os.path.join(dest_dir, 'top_' + str(NUM_TOP_i_IMAGES) + '_distributions.txt')

    logger.info('Start to write dictionary')
    with open(dict_filename, 'w') as f:
        for i, (filename, distribution) in tqdm(enumerate(given_dict.items())):
            if i >= NUM_TOP_i_IMAGES:
                break
            f.write('{}. {}\n'.format(i+1, filename))
            f.write('\tProbabilities\tCategories\n')
            prob_sum = 0
            for j, (cate, score) in enumerate(distribution):
                if j > NUM_TOP_p_Predictions:
                    break
                prob_sum = prob_sum + score
                f.write('\t{0:.5f}:  {1}\n'.format(score, cate))
            f.write('\t{0:.5f}:  SUM\n'.format(prob_sum))
    logger.info('Finish writing dictionary')

def ambiguous_copy_n_write(given_dict: dict, dataset_name: str, type_: str):
    """ Copy images and Write distribution file """
    ambiguous_copy_files(given_dict, dataset_name, type_)
    ambiguous_write_dict(given_dict, dataset_name, type_)

def get_dict_with_top_k_pairs(sorted_dict, top_k):
    """ This is for testing
        Store as 
              ('n0001.jpg',
               [('dog', 0.8), ('cat', 0.1), ...])
               ---------- top_k pairs ----------
    """
    result = []
    for key, cate_score_pairs in sorted_dict.items():
        result.append((key, cate_score_pairs[:top_k]))
    return result

if __name__ == '__main__':
    from pprint import pprint
    from classify_images import get_predictions_dict
    
    predictions_dict = get_predictions_dict('fall11_urls_top30')

    sorted_by_confidence_dict = find_confident_images(predictions_dict)
    # pprint(get_dict_with_top_k_pairs(sorted_by_confidence_dict, 2))
    # print(len(sorted_by_confidence_dict))
    ambiguous_copy_n_write(sorted_by_confidence_dict, 'fall11_urls_top30', 'confident')
    print("-"*40)

    sorted_by_ambiguous_dict = find_ambiguous_images(predictions_dict, 2, 60)
    # pprint(get_dict_with_top_k_pairs(sorted_by_ambiguous_dict, 2))
    # print(len(sorted_by_ambiguous_dict))
    ambiguous_copy_n_write(sorted_by_ambiguous_dict, 'fall11_urls_top30', 'ambiguous')