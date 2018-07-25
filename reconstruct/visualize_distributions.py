
import os
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import DIR, TARGET_DIR, OUTPUT_DIR, NUM_TOP_i_IMAGES

import daiquiri as dqr 
import logging

dqr.setup(level=logging.DEBUG)
logger = dqr.getLogger()

def combine_distribs_n_select_top_10(target_distrib: list, dataset_distrib: list):
    """ Given two distributions, we combine them and select 10 categories.

    This function is aimed to find top 10 most activated categories for both distributions.
    First, we sort each distribution seperately, by scores. Now we have two lists of distributions as 
    sorted_target_distrib:  t_pair1 -> t_pair2 -> t_pair3 -> ...
    sorted_dataset_distrib: t_pair1 -> t_pair2 -> t_pair3 -> ...

    Args:
        target_distrib: [(cate1, score1), (cate2, score2), ...]
        dataset_distrib: [(cate1, score1), (cate2, score2), ...]
    Return:
        list: indices of selected categories
    """
    indexed_target_distrib  = [(i, cate, score) for i, (cate, score) in enumerate(target_distrib)]
    indexed_dataset_distrib = [(i, cate, score) for i, (cate, score) in enumerate(dataset_distrib)]
    combined_distrib = sorted(indexed_target_distrib + indexed_dataset_distrib, key=lambda pair: pair[2], reverse=True)

    selected_indices = []
    for (i, _, _) in combined_distrib:
        if not i in selected_indices:
            if len(selected_indices) > 10:
                break
            selected_indices.append(i)
    return sorted(selected_indices)

def visualize_distribution(sorted_sim_dict: dict, dataset_name: str, type_: str):
    """ Visualize the the distributions in dataset instance and target instance
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

    logger.info('Start visualizing distributions.')
    for (target_filename, (target_distrib, sorted_target_sim_dict)) in tqdm(sorted_sim_dict.items()):
        tar_dest_dir = os.path.join(dest_dir, target_filename)

        # We only write one png file per target_instance
        _, sub = plt.subplots(NUM_TOP_i_IMAGES)
        logger.debug('Finish initializing subplots')

        logger.info('Start specifing each dataset instance for {}'.format(target_filename))
        # Dataset 
        for i, (dataset_filename, (dataset_distrib, _)) in enumerate(sorted_target_sim_dict.items()):
            if i >= NUM_TOP_i_IMAGES:
                break
            # Find indices of selected top 10 categories
            selected_indices = combine_distribs_n_select_top_10(target_distrib, dataset_distrib)

            x = [target_distrib[i][0] for i in selected_indices]
            y_tar = [target_distrib[i][1]  for i in selected_indices]
            y_dst = [dataset_distrib[i][1] for i in selected_indices]
            
            sub[i].barh(x, y_tar, alpha=0.5, color='b', label=target_filename)
            sub[i].barh(x, y_dst, alpha=0.5, color='r', label=dataset_filename)
            sub[i].legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 5})
            sub[i].tick_params(axis='both', which='major', labelsize=5)
        # Plot distributions into tar_dest_dir
        logger.debug('Will save to {}'.format(os.path.join(
            tar_dest_dir, target_filename.split('.')[0] + '_distribion_compares.png')))
        # Configure plot layout
        plt.tight_layout(pad=0.2)
        plt.savefig(os.path.join(
            tar_dest_dir, target_filename.split('.')[0] + '_distribion_compares.png'), bbox_inches='tight', dpi=300)
        logger.debug('Finish saving')



if __name__ == '__main__':

    # Set dataset_name
    dataset_name = sys.argv[1]
    # Set target_folder_name
    target_folder_name = sys.argv[2]

    from download_inceptionV3 import maybe_download_and_extract
    maybe_download_and_extract()

    from find_similar_images import maybe_sample_images, get_similarity_dict, get_sorted_similarity_dict, similar_copy_files, similar_write_files
    from classify_images import run_predictions, get_predictions_dict

    # Get dataset and target dictionaries
    dataset_dict = get_predictions_dict(os.path.join(DIR, dataset_name))
    target_dict = get_predictions_dict(os.path.join(TARGET_DIR, target_folder_name))

    # Use different similarity algorithm to find similar images
    for sim_algo in ['COS', 'KL']:
        sim_dict = get_similarity_dict(target_dict, dataset_dict, sim_algo)
        sorted_sim_dict = get_sorted_similarity_dict(sim_dict)

        # Visualize distributions
        visualize_distribution(sorted_sim_dict, dataset_name, sim_algo)
