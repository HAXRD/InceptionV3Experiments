import os
import sys

from download_inceptionV3 import maybe_download_and_extract
from classify_images import get_predictions_dict
from find_ambiguous_images import find_confident_images, find_ambiguous_images, ambiguous_copy_n_write
from find_similar_images import maybe_sample_images, get_similarity_dict, get_sorted_similarity_dict, similar_copy_files, similar_write_files
from visualize_distributions import visualize_distribution
from config import DIR, TARGET_DIR

if __name__ == '__main__':

    # Set dataset_name
    dataset_name = sys.argv[1]
    # Set target_folder_name
    target_folder_name = sys.argv[2]

    # Download and extract model tar files
    maybe_download_and_extract()

    # Maybe sample some target images
    maybe_sample_images(dataset_name)

    # Get dataset_dict and target_dict
    dataset_dict = get_predictions_dict(os.path.join(DIR, dataset_name))
    target_dict  = get_predictions_dict(os.path.join(TARGET_DIR, target_folder_name))

    # Get most confident and ambiguous images
    sorted_by_confidence_dict = find_confident_images(dataset_dict)
    sorted_by_ambiguous_dict = find_ambiguous_images(dataset_dict, 2, 60)
    ambiguous_copy_n_write(sorted_by_confidence_dict, dataset_name, 'confident')
    ambiguous_copy_n_write(sorted_by_ambiguous_dict,  dataset_name, 'ambiguous')

    # Use different similarity algorithm to find similar images
    for sim_algo in ['COS', 'KL']:
        sim_dict = get_similarity_dict(target_dict, dataset_dict, sim_algo)
        sorted_sim_dict = get_sorted_similarity_dict(sim_dict)

        similar_copy_files(sorted_sim_dict, dataset_name, sim_algo)
        similar_write_files(sorted_sim_dict, dataset_name, sim_algo)

        # Visualize distributions
        visualize_distribution(sorted_sim_dict, dataset_name, sim_algo)