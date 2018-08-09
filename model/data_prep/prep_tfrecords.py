"""
Author: Xu Chen
Email: xu.chen at dal.ca
"""

import sys
import os
import glob
import tensorflow as tf 
import numpy as np 

import matplotlib.pyplot as plt

import logging 
import daiquiri

daiquiri.setup(level=logging.INFO)
logger = daiquiri.getLogger(__name__)

import random

random.seed(1234)

import cv2

TRAIN_RATIO = 0.8

TRAIN_FILENAME = 'train.tfrecord'
TEST_FILENAME  = 'test.tfrecord'

NEG_DIR = './negative'
POS_DIR = './positive'
# Store images in generating process, just for verification purpose
GEN_DIR = './generating'
# Restore images in testing process, just for verification purpose
OUT_DIR = './output'

H, W = 299, 299

def write_data_to_tfrecord(neg_dataset_name: str, pos_dataset_name: str, shuffle: bool=True):
    """ Write image files into tfrecords. 
    
    Args:
        neg_dataset_name: negative dataset folder name,
        pos_dataset_name: positive dataset folder name
    Credit:
        How to write into and read from a TFRecords file in TensorFlow: 
        http://machinelearninguru.com/deep_learning/data_preparation/tfrecord/tfrecord.html
    """
    
    # Define negative class and positive class image paths
    neg_path = os.path.join(NEG_DIR, neg_dataset_name, '*')
    pos_path = os.path.join(POS_DIR, pos_dataset_name, '*')
    logger.info(f"Negative images directory: {os.path.join(NEG_DIR, neg_dataset_name)}")
    logger.info(f"Positive images directory: {os.path.join(POS_DIR, pos_dataset_name)}")

    from time import time
    start = time()

    """ Read data """
    neg_addrs = glob.glob(neg_path)
    pos_addrs = glob.glob(pos_path)
    neg_labels = [0 for _ in neg_addrs]
    pos_labels = [1 for _ in pos_addrs]
    logger.info(f"Negative dataset size={len(neg_addrs)}")
    logger.info(f"Positive dataset size={len(pos_addrs)}")

    # We want our data to be evenly distributed for training and testing set
    #   So we shuffle the negative and positive images seperatly first,
    #   then after joining them, we shuffle the order again.
    neg_data = list(zip(neg_addrs, neg_labels))
    pos_data = list(zip(pos_addrs, pos_labels))
    if shuffle == True:
        random.shuffle(neg_data)
        random.shuffle(pos_data)

    train_data = neg_data[:int(TRAIN_RATIO*len(neg_data))] + pos_data[:int(TRAIN_RATIO*len(pos_data))]
    test_data  = neg_data[int(TRAIN_RATIO*len(neg_data)):] + pos_data[int(TRAIN_RATIO*len(pos_data)):]
    if shuffle == True:
        random.shuffle(train_data)
        random.shuffle(test_data)
    
    """ Write images to tfrecords """
    serialize_images_to_tfrecord(TRAIN_FILENAME, train_data)
    serialize_images_to_tfrecord(TEST_FILENAME,  test_data)
    logger.info(f"Done writing '{TRAIN_FILENAME}' and '{TEST_FILENAME}'. Total time: {round(time() - start, 2)}s.")

def serialize_images_to_tfrecord(tfrecord_filename: str, data: list):
    """ Write images to tfrecord

    Args:
        tfrecord_filename = specify filename,
        data = a list of (image_path, image_label) tuples
    """
    
    # Get image paths and image labels
    addrs, labels = zip(*data)
    logger.info(f"{tfrecord_filename} len: {len(addrs)}")

    # Create TFRecordWriter and get the phase type
    writer = tf.python_io.TFRecordWriter(tfrecord_filename)
    phase_type = tfrecord_filename.split('.')[0] # 'train' or 'test'

    for i in range(len(addrs)):
        # Log for every 100 images
        if (i+1) % 100 == 0 and not i == 0:
            logger.info(f"Write {phase_type} images {i+1}")

        # Load the image and label
        img, img_shape = load_image(addrs[i])
        logger.debug(img_shape)
        label = labels[i]


        # Create a feature
        feature = {f'{phase_type}/label': _int64_feature(label),
                   f'{phase_type}/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
        
        # Create an example
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
    
    # Close the writer
    writer.close()
    sys.stdout.flush()

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def load_image(addr: str, Write: bool=False):
    """ Read an image with its address (path)
        
    Args:
        addr = relative path of the image
    Returns:
        img = image
    """
    img = cv2.imread(addr)
    try:
        img = cv2.resize(img, (H, W), interpolation=cv2.INTER_CUBIC)
    except Exception as e:
        logger.error(f'{addr} error: {e}')
    # Write image for testing purpose
    if Write:
        if not os.path.exists(GEN_DIR):
            os.makedirs(GEN_DIR)
        cv2.imwrite(os.path.join(GEN_DIR, os.path.basename(addr)), img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)

    return img, img.shape


def read_tfrecords(tfrecord_filename: str, epochs: int):
    """ Read tfrecord and produce given number of epochs 

    Args:
        tfrecord_filename = 'train.tfrecord' or 'test.tfrecord',
        epochs = number of epoch we want to produce for data
    Returns:
        img, label = image and corresponding label

    Credit:
        http://ycszen.github.io/2016/08/17/TensorFlow高效读取数据/
        http://machinelearninguru.com/deep_learning/data_preparation/tfrecord/tfrecord.html
    """
    
    assert os.path.exists(tfrecord_filename) == True
    logger.debug(tfrecord_filename)

    # Get phase type
    phase_type = tfrecord_filename.split('.')[0]

    # Create a list of filenames and write_data_to_tfrecord('', 'imgs_520') it into a FIFO queue
    filename_queue = tf.train.string_input_producer([tfrecord_filename], num_epochs=epochs)
    
    # Create a tfrecord reader
    reader = tf.TFRecordReader()

    # Read queue. It will return a filename and a file
    _, serialized_example = reader.read(filename_queue)
    # Get features from one example
    features = tf.parse_single_example(serialized_example,
                                       features={f'{phase_type}/image': tf.FixedLenFeature([], tf.string),
                                                 f'{phase_type}/label': tf.FixedLenFeature([], tf.int64)})
    # Decode image from binary format
    img = tf.decode_raw(features[f'{phase_type}/image'], tf.float32)

    # Reshape image to its original shape
    img = tf.reshape(img, [H, W, 3])
    # Cast label data into int32 to save memory
    label = tf.cast(features[f'{phase_type}/label'], tf.int32)

    return img, label

def run_test(tfrecord_filename: str, epochs: int=2, write: bool=False):
    """ A testing code for reading TFRecords we generated
    
    Args:
        tfrecord_filename = 'train.tfrecord' or 'test.tfrecord'
    """
    
    assert os.path.exists(tfrecord_filename) == True

    image, label = read_tfrecords(tfrecord_filename, epochs)

    # Creates batches by randomly shuffling tensors
    batch_size = 8
    images, labels = tf.train.shuffle_batch([image, label],
                                            batch_size=batch_size,
                                            capacity=batch_size * 64,
                                            min_after_dequeue=batch_size * 32,
                                            allow_smaller_final_batch=False)
    logger.debug(f'images shape: {images.get_shape()}, labels shape: {labels.get_shape()}')

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(init)

        # Start queue with coordinator
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)
        logger.info('Start running queue...')
        
        counter = 1
        try:
            while not coordinator.should_stop():
                img, lbl = sess.run([images, labels])
                print(counter, img.shape, lbl)
                
                if write == True:
                    if not os.path.exists(OUT_DIR):
                        os.makedirs(OUT_DIR)
                    for j in range(batch_size):
                        cv2.imwrite(os.path.join(OUT_DIR, f"{counter}_{j}.jpg"), img[j, ...])

                counter = counter + 1

        except Exception as e:
            coordinator.request_stop(e)

        coordinator.join(threads)

    logger.debug('Test read tf record succeed')

if __name__ == '__main__':
    
    kind = sys.argv[1]

    if kind == 'Produce':
        write_data_to_tfrecord('fall11_urls_top30', 'imgs', False)
    elif kind == 'for_testing':
        write_data_to_tfrecord('for_testing', 'for_testing', False)
    elif kind == 'prod':
        write_data_to_tfrecord('fall11_urls_top10000_seed1', 'imgs_500')
    else:
        run_test(TRAIN_FILENAME, True)


    
    