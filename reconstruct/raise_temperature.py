import sys
import os
from config import TARGET_DIR
import numpy as np 

def raise_temperature(dict_: dict, T: float=10):
    """
    Args:
        dict_: 
        T:      
    """
    epsilon = 1e-10
    smoothed_dict = {}
    for image_name, readable_predictions in dict_.items():
        # store human strings
        human_strings = [human_string 
                         for (human_string, _) in readable_predictions]
        # store scores
        np_scores = np.array([score 
                          for (_, score) in readable_predictions])
        np_exp_scores = np.exp(np_scores/T)
        # raise temperatures for each score
        np_scores_sum = np.sum(np_exp_scores)
        np_smoothed_scores = np.divide((np_exp_scores), (np_scores_sum))
        smoothed_scores = np_smoothed_scores.tolist()
    
        smoothed_readable_predictions = [(human_string, score)
                                         for human_string, score in zip(human_strings, smoothed_scores)]

        smoothed_dict[image_name] = smoothed_readable_predictions
    return smoothed_dict

if __name__ == '__main__':
    # Set dataset_name
    dataset_name = sys.argv[1]
    # Set target_folder_name
    target_folder_name = sys.argv[2]

    from download_inceptionV3 import maybe_download_and_extract
    # Download and extract model tar files
    maybe_download_and_extract()

    from find_similar_images import maybe_sample_images
    # Maybe sample some target images
    maybe_sample_images(dataset_name)

    from classify_images import get_predictions_dict
    # Get dataset_dict and target_dict
    target_dict  = get_predictions_dict(os.path.join(TARGET_DIR, target_folder_name))
    # print(target_dict)
    
    smoothed_target_dict = raise_temperature(target_dict)
    # print(smoothed_target_dict)