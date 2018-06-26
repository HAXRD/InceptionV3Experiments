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

def maybe_download_images_with_urls(limit=10):
    """Read the image_urls file and download the images with urls"""
    # IMAGE_URLS = 'http://image-net.org/imagenet_data/urls/imagenet_fall11_urls.tgz'
    dest_directory = FLAGS.model_dir # /tmp/inception
    foldername = 'sample_1000_images'
    folderpath = os.path.join(dest_directory, foldername)
    print('Images are saved %s' % (folderpath))

    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
    with open(os.path.join(dest_directory, 'fall11_urls.txt'), 'r') as f:
        num = 0
        for line in f:
            if num > limit:
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

def main(_):
    maybe_download_and_extract(IMAGE_URLS)
    maybe_download_images_with_urls(FLAGS.dataset_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_dir',
        type=str,
        default='/tmp/inception',
        help='Absolute directory to store the model files.'
    )
    parser.add_argument(
        '--dataset_size',
        type=int,
        default=1000,
        help='The size of image dataset.'
    )
    # random mode has not been implemented yet
    parser.add_argument(
        '--read_mode',
        type=str,
        default='default',
        help='Mode for reading txt file, default or random.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)