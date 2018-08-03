import os
import sys
import re
import argparse
import shutil

SRC_DIR = '../../reconstruct/output'
DATA_DIR = '../../reconstruct/data'
NEG_DIR = './negative'

def prepare_negative_dataset(dataset_name, method_name, src_dataset_name):

    directory = os.path.join(SRC_DIR, dataset_name, 'similar', method_name)

    # Get all the folders in the directory
    folders = [folder for folder in os.listdir(directory)
               if os.path.isdir(os.path.join(directory, folder))]

    neg_dataset = set()

    for folder in folders:
        # Get all the files in folder
        files = [file for file in os.listdir(os.path.join(directory, folder))
                 if os.path.isfile(os.path.join(directory, folder, file))]

        # Create regular expression to match target image file
        neg_matcher = re.compile(r'\d*_.*\..*')
        for file in files:
            filename = os.path.basename(file) 
            if neg_matcher.match(filename):
                # Remove prefix
                filename = '_'.join(filename.split('_')[1:])
                # Add to negative_dataset
                neg_dataset.add(filename)

    # Convert set to list
    neg_dataset = list(neg_dataset)    
    print(neg_dataset)

    # Create folder if not exist
    output_dir = os.path.join(NEG_DIR, src_dataset_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in neg_dataset:
        file_path = os.path.join(DATA_DIR, src_dataset_name, file)
        print(file_path)

        if os.path.exists(file_path):
            shutil.copyfile(file_path, os.path.join(output_dir, file))
        

    # print(folders)


if __name__ == '__main__':
    
    # Access directory: ../../reconstruct/output/dataset_name/similar/method_name
    dataset_name = sys.argv[1]
    method_name = sys.argv[2]
    src_dataset_name = sys.argv[3]

    prepare_negative_dataset(dataset_name, method_name, src_dataset_name)

