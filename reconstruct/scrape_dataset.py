"""
This program is aimed to fetch the sample dataset from ImageNet,
'imagenet_fall11_urls.tgz' specificly. After downloaded the url
txt file, it reads the lines and download the given amount siz of
images (either the first siz images or the randomly siz images).
"""

import sys
import os
import re
import random
import argparse
from six.moves import urllib

import tensorflow as tf 

from config import DIR, URL_FILE
import daiquiri as dqr 
import logging

dqr.setup(level=logging.DEBUG)
logger = dqr.getLogger()

FLAGS = None

dev = {
    'upper':     30,
    'read_mode': 'DEFAULT',
    'seed':      5
}
prod = {
    'upper':     1000,
    'read_mode': 'DEFAULT',
    'seed':      5
}



def initial_urls_file_path():
    """ Initialize configured 'url_file' path, because we are 
        not going to download all the images from 'fall11_urls.txt'"""
    if FLAGS.read_mode == 'DEFAULT':
        path = os.path.join(
            DIR, URL_FILE.split('.')[0] + '_top' + str(FLAGS.upper) + '.txt')
        return path
    else:
        path = os.path.join(
            DIR, URL_FILE.split('.')[0] + '_top' + str(FLAGS.upper) + '_seed' + str(FLAGS.seed) + '.txt')
        return path

def write_url_file(filepath: str):
    """Write smaller size of 'write_url_file' according to user's args"""

    if not os.path.exists(filepath):
        if FLAGS.read_mode == 'DEFAULT':
            logger.info('{}: start reading {}'.format(FLAGS.read_mode, os.path.join(DIR, URL_FILE)))
            data = []
            with open(os.path.join(DIR, URL_FILE), 'rb') as source:
                for i, line in enumerate(source):
                    if i >= FLAGS.upper:
                        break
                    data.append(line)
            logger.info('{}: start writing {}'.format(FLAGS.read_mode, filepath))
            with open(filepath, 'wb') as target:
                for line in data:
                    target.write(line)
            logger.info('{}: finish writing {}'.format(FLAGS.read_mode, filepath))
        else: # RANDOM
            # Set seed
            random.seed(FLAGS.seed)
            logger.info('{}: start reading {}'.format(FLAGS.read_mode, os.path.join(DIR, URL_FILE)))
            data = []
            with open(os.path.join(DIR, URL_FILE), 'rb') as source:
                data = sorted([(random.random(), line) for line in source], key=lambda x: x[1])
            logger.info('{}: start writing {}'.format(FLAGS.read_mode, filepath))
            with open(filepath, 'wb') as target:
                for i, (_, line) in enumerate(data):
                    if i >= FLAGS.upper:
                        break
                    target.write(line)
                    
            logger.info('{}: finish writing {}'.format(FLAGS.read_mode, filepath))
        pass
    else:
        logger.warning('File {} already exists!'.format(os.path.basename(filepath)))

def download_images(filepath: str):
    """Read the 'url_file' and download the images with urls."""

    dest_dir = os.path.join(DIR, os.path.basename(filepath).split('.')[0])
    logger.debug('dest_dir: {}'.format(dest_dir))
    
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    logger.info('Images will be stored in {}'.format(dest_dir))
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            image_name, image_url = line.split()

            # Filter out invalid images
            basename = os.path.basename(image_url)
            if 'jpg' in basename:
                image_name = image_name + '.jpg'
            elif 'png' in basename:
                image_name = image_name + '.png'
            else:
                continue

            req = urllib.request.Request(image_url)
            try:
                urllib.request.urlopen(req, timeout=1)
                urllib.request.urlretrieve(
                    image_url, os.path.join(dest_dir, image_name))
                logger.debug(
                    '>> Downloaded {0} | {1:.1f}%'.format(image_name, float(i+1)/float(FLAGS.upper)*100.))
            except:
                logger.error('>> Error occurred: {} {}'.format(image_name, image_url))
    logger.info('Finish downloading images. Images are stored in {}'.format(dest_dir))
    logger.info('Dataset_name: {}'.format(os.path.basename(dest_dir)))

def main(_):
    download_file_path = initial_urls_file_path()
    logger.debug('Target file path: {}'.format(download_file_path))

    write_url_file(download_file_path)
    download_images(download_file_path)
    

if __name__ == '__main__':
    
    curr = dev
    # curr = prod

    parser = argparse.ArgumentParser()
    parser.add_argument('--upper', type=int, default=curr['upper'],
                        help="Upperbound size for image dataset.")
    parser.add_argument('--read_mode', type=str, default=curr['read_mode'],
                        help="Whether read file by default order or random order.")
    parser.add_argument('--seed', type=int, default=curr['seed'],
                        help="Seed number if use random read order.")
    
    FLAGS, _ = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]])

