#!/bin/sh
python find_similar_images_with_distribution.py --dataset_name sample_fall11_urls_t10000_s1 
python find_similar_images_with_distribution.py --dataset_name sample_fall11_urls_t10000_s1 --dict_mode RESTORE --similarity_method KL

python find_similar_images_with_distribution.py --dataset_name sample_fall11_urls_t10000_s5 
python find_similar_images_with_distribution.py --dataset_name sample_fall11_urls_t10000_s5 --dict_mode RESTORE --similarity_method KL
