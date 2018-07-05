"""
This program is aimed to fetch the sample dataset from ImageNet,
'imagenet_fall11_urls.tgz' specificly. After downloaded the url
txt file, it reads the lines and download the given amount siz of
images (either the first siz images or the randomly siz images).
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import argparse
import os.path
import re
import sys
import tarfile
import wget
import random

import numpy as np
from six.moves import urllib
import tensorflow as tf

FLAGS = None

IMAGE_URLS = 'http://image-net.org/imagenet_data/urls/imagenet_fall11_urls.tgz'

def maybe_download_and_extract(URL):
    """Download and extract tar file."""
    dest_directory = FLAGS.model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)
    else:
        print('{} exist, jumping download_and_extract'.format(filepath))

def maybe_download_images_with_urls(path):
    """Read the image_urls file and download the images with urls"""
    # IMAGE_URLS = 'http://image-net.org/imagenet_data/urls/imagenet_fall11_urls.tgz'
    dest_directory = FLAGS.model_dir # /tmp/inception
    foldername = 'sample_' + path.split('/')[-1].split('.')[0]
    folderpath = os.path.join(dest_directory, foldername)
    print('Images are saved %s' % (folderpath))

    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
    with open(path, 'r') as f:
        num = 0
        for line in f:
            if num > FLAGS.dataset_size:
                break
            [image_name, image_url] = line.split()
            extension = image_url.split('.')[-1]
            if 'jpg' in extension:
                extension = 'jpg'
            elif 'png' in extension:
                extension = 'png'
            else:
                continue
            # print(line.split())
            req = urllib.request.Request(image_url)
            try:
                urllib.request.urlopen(req, timeout=1)
                def _progress(count, block_size, total_size):
                    percentage = float(count * block_size) / float(total_size + 1) * 100.0
                    sys.stdout.write('\r>> %d Downloading %s %.1f%%' % (
                        num+1, image_name, percentage if percentage <= 100 else 100))
                # print(image_url, os.path.join(folderpath, image_name + '.' + extension))        
                urllib.request.urlretrieve(image_url, os.path.join(folderpath, image_name + '.' + extension), _progress)
                print()
            except urllib.error.HTTPError as e:
                print('HTTP Error: ', e.reason, '\t', image_url)
            except urllib.error.URLError as e:
                print('URL Error: ', e.reason, '\t', image_url)
            except:
                print('Unknown Error: \t', image_url)
            num = num + 1

def initial_urls_file_path():
    path = None
    if FLAGS.read_mode == 'random':
        path = os.path.join(FLAGS.model_dir, 'fall11_urls_t'+str(FLAGS.dataset_size)+'_s'+str(FLAGS.seed)+'.txt')
    else: 
        path = os.path.join(FLAGS.model_dir, 'fall11_urls_t'+str(FLAGS.dataset_size)+'.txt')
    # print(path)
    return path

def maybe_shuffle_urls_file(path):
    if not os.path.exists(path):
        # Set seed
        random.seed(FLAGS.seed)
        # Read file
        with open(os.path.join(FLAGS.model_dir, 'fall11_urls.txt'), 'r') as source:
            data = [(random.random(), line) for line in source]
        print("Finish reading...")
        if FLAGS.read_mode == 'random':
            data.sort()
        print("Writing to file: ", path)
        with open(path, 'w') as target:
            cnt = 0
            for _, line in data:
                if cnt > FLAGS.dataset_size:
                    break
                target.write(line)
                cnt = cnt + 1
    else:
        print('File exist, jumping sorting step!')

def main(_):
    maybe_download_and_extract(IMAGE_URLS)
    download_file_path = initial_urls_file_path()
    maybe_shuffle_urls_file(download_file_path)
    maybe_download_images_with_urls(download_file_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    dev_mac = {
        'model_dir':    '/tmp/inception',
        'dataset_size': 10,
        'read_mode':    'default',
        'seed':         5
    }
    dev_ubu = {
        'model_dir':    '/tmp/inception',
        'dataset_size': 100,
        'read_mode':    'default',  # specify this only in terminal
        'seed':         5
    }
    prod_ubu = {
        'model_dir':    '/home/xu/Documents/inception',
        'dataset_size': 10000,
        'read_mode':    'default',  # specify this only in terminal
        'seed':         5
    }

    # use_dict = dev_mac
    # use_dict = dev_ubu
    use_dict = prod_ubu

    parser.add_argument(
        '--model_dir',
        type=str,
        default=use_dict['model_dir'],
        help='Absolute directory to store the model files.'
    )
    parser.add_argument(
        '--dataset_size',
        type=int,
        default=use_dict['dataset_size'],
        help='The size of image dataset.'
    )
    # random mode has not been implemented yet
    parser.add_argument(
        '--read_mode',
        type=str,
        default=use_dict['read_mode'],
        help='Mode for reading txt file, default or random.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=use_dict['seed'],
        help="Random seed, default=5."
    )
    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)