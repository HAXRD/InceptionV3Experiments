python find_similar_images_with_distribution.py \
--model_dir /home/xu/Documents/inception    \
--dataset_name testing_images   \
--sort_method default   \
--num_top_i_images 10   \
--output_dir /home/xu/Documents/output 



python find_similar_images_with_distribution.py \
--dict_mode SAVE \
--model_dir /home/xu/Documents/inception    \
--dataset_name sample_1000_images   \
--sort_method default   \
--num_top_i_images 10   \
--output_dir /home/xu/Documents/output 



python find_similar_images_with_distribution.py \
--dict_mode RESTORE \
--model_dir /home/xu/Documents/inception    \
--dataset_name sample_1000_images   \
--sort_method default   \
--num_top_i_images 10   \
--output_dir /home/xu/Documents/output 
--target_dir /home/xu/Documents/target 