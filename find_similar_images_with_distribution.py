# coding: utf-8


"""Find most indecisive images from given set of images.

Run with Inception trained on ImageNet 2012 Challenge data
set.

This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. It outputs human readable
strings of the top 5 predictions along with their probabilities.

Reference:
  https://github.com/tensorflow/models

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import re
import sys
import tarfile
import pprint
import collections
import shutil
from tqdm import tqdm
import pickle

import numpy as np
from six.moves import urllib
import tensorflow as tf

FLAGS = None

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long

# Reuse the nodelookp mapping class from classify_image.py
class NodeLookup(object):
    """Converts integer node ID's to human readable labels."""

    def __init__(self,
                 label_lookup_path=None,
                 uid_lookup_path=None):
        if not label_lookup_path:
            label_lookup_path = os.path.join(
                    FLAGS.model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
        if not uid_lookup_path:
            uid_lookup_path = os.path.join(
                    FLAGS.model_dir, 'imagenet_synset_to_human_label_map.txt')
        self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

    def load(self, label_lookup_path, uid_lookup_path):
        """Loads a human readable English name for each softmax node.

        Args:
            label_lookup_path: string UID to integer node ID.
            uid_lookup_path: string UID to human-readable string.

        Returns:
            dict from integer node ID to human-readable string.
        """
        if not tf.gfile.Exists(uid_lookup_path):
            tf.logging.fatal('File does not exist %s', uid_lookup_path)
        if not tf.gfile.Exists(label_lookup_path):
            tf.logging.fatal('File does not exist %s', label_lookup_path)

        # Loads mapping from string UID to human-readable string
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        p = re.compile(r'[n\d]*[ \S,]*')
        for line in proto_as_ascii_lines:
            parsed_items = p.findall(line)
            uid = parsed_items[0]
            human_string = parsed_items[2]
            uid_to_human[uid] = human_string

        # Loads mapping from string UID to integer node ID.
        node_id_to_uid = {}
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        for line in proto_as_ascii:
            if line.startswith('  target_class:'):
                target_class = int(line.split(': ')[1])
            if line.startswith('  target_class_string:'):
                target_class_string = line.split(': ')[1]
                node_id_to_uid[target_class] = target_class_string[1:-2]

        # Loads the final mapping of integer node ID to human-readable string
        node_id_to_name = {}
        for key, val in node_id_to_uid.items():
            if val not in uid_to_human:
                tf.logging.fatal('Failed to locate: %s', val)
            name = uid_to_human[val]
            node_id_to_name[key] = name

        return node_id_to_name

    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]

def progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='▊'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration//total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    if iteration == total:
        print()

def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(os.path.join(
            FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
        

def run_predictions(images_dir):
    """Runs predictions on a list of images.

    Args:
        image_dir: directory of given image(s).

    Returns:
        A dictionary of top k category-score pairs.
    """
    # Check if directory is valid
    if not tf.gfile.IsDirectory(images_dir):
        tf.logging.fatal('Folder does not exist %s', images_dir)

    image_top_k_category_score_pair_dictionary = {}

    category_score_pair_dictionary = {}

    # Creates node ID --> English string lookup.
    node_lookup = NodeLookup()

    # Creates graph from saved GraphDef.
    create_graph()

    with tf.Session() as sess:
        # Some useful tensors:
        # 'softmax:0': A tensor containing the normalized prediction across
        #   1000 labels.
        # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
        #   float description of the image.
        # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
        #   encoding of the image.
        # Runs the softmax tensor by feeding the image_data as input to the graph.
        softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')

        total_length = len(tf.gfile.ListDirectory(images_dir))
        # Run predictions for a list of images and store them in dictionary as category_score_pair_dictionary
        for idx, image_name in enumerate(tf.gfile.ListDirectory(images_dir)):
            image_data = tf.gfile.FastGFile(os.path.join(images_dir, image_name), 'rb').read()
            predictions = sess.run(softmax_tensor,
                                  {'DecodeJpeg/contents:0': image_data})
            predictions = np.squeeze(predictions)
            # print('predictions:', predictions)
            # print('shape:      ', predictions.shape)

            # top_k = predictions.argsort()[:][::-1]
            # print('top_k:', top_k)
            # print('shape:      ', top_k.shape)

            category_score_pairs = []
            for node_id, score in enumerate(predictions):
                human_string = node_lookup.id_to_string(node_id)
                category_score_pairs.append((human_string,score))
            category_score_pair_dictionary[image_name] = category_score_pairs
            progress_bar(idx + 1, total_length, prefix='Classification process', suffix='Completed.', length=50)
    return category_score_pair_dictionary


def maybe_download_and_extract():
    """Download and extract model tar file."""
    dest_directory = FLAGS.model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                    filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def sort_dictionary(given_dictionary, method='default'):
    # Sort the dic with 'FLAGS.sort_method' as sorted_dict:
    #   1. sort the predictions, and only store the top 'FLAGS.num_top_p_predictions'.
    predictions_sorted_dictionary = {}
    
    for key in given_dictionary:
        predictions_sorted_dictionary[key] = sorted(given_dictionary[key], key=lambda x: x[1], reverse=True)[:FLAGS.num_top_p_predictions]

    #   2. sort the files according to its indecisiveness and only store the top 'FLAGS.num_top_i_images'.
    if (method == 'weighted'):
        def weighted_lambda(item):
            weighted_sum = 0
            mean = sum([item[1][i][1] for i in range(FLAGS.num_top_p_predictions)])/FLAGS.num_top_p_predictions
            for i in range(FLAGS.num_top_p_predictions):
                curr_score = item[1][i][1]
                weighted_sum = weighted_sum + (curr_score - mean)**2
            return weighted_sum
        return collections.OrderedDict(sorted(predictions_sorted_dictionary.items(), key=weighted_lambda))
    else: # default
        return collections.OrderedDict(sorted(predictions_sorted_dictionary.items(), 
                                    key=lambda it: it[1][0][1], reverse=True))

def sort_similarity_dictionary(sim_dic):
    sorted_sim_dic = {}
    for key, item in sim_dic.iteritems():
        sorted_sim_dic[key] = sorted(item, key=lambda x: x[1], reverse=True)[:FLAGS.num_top_s_similar]
    return sorted_sim_dic

def write_to_file(given_dictionary, directory, filename):
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(os.path.join(directory, filename), 'w') as f:
        total_length = len(given_dictionary)
        for idx, (key, item) in tqdm(enumerate(given_dictionary.iteritems())):
            f.write('{}. {}\n'.format(idx, key))
            for pair in item:
                f.write('\t%.9f:  %s\n' % (pair[1], pair[0]))
            # progress_bar(idx + 1, total_length, prefix='Writing progress ', suffix='Completed', length=50)

def filter_copy_files_to_dir(given_dictionary, directory, mode):
    # Clean the directory
    if os.path.exists(directory):
        shutil.rmtree(directory, ignore_errors=True)
    os.makedirs(directory)
    
    for key, item in tqdm(enumerate(given_dictionary.iteritems())):
        if mode == 0:
            filename = item[0]
            shutil.copyfile(os.path.join(FLAGS.model_dir, FLAGS.dataset_name, filename),
                                     os.path.join(directory, str(key) + '.' + filename.split('.')[-1]))
        elif mode == 1:
            for i, pair in enumerate(item[1]):
                print(pair)
                filename = pair[0]
                shutil.copyfile(os.path.join(FLAGS.model_dir, FLAGS.dataset_name, filename),
                                     os.path.join(directory, str(i) + '.' + filename.split('.')[-1]))


def calculate_log_similarity(tar_item, std_item):
    tar_vec = np.array([tup[1] for tup in tar_item]) 
    std_vec = np.array([tup[1] for tup in std_item])

    dot = np.dot(tar_vec, std_vec)
    mod_tar_vec = np.sqrt(np.sum(np.square(tar_vec)))
    mod_std_vec = np.sqrt(np.sum(np.square(std_vec)))
    similarity = dot/(mod_tar_vec*mod_std_vec)
    return np.log10(similarity)

def main(_):
    # Download checkpoint files and etc.
    maybe_download_and_extract()
    
    dic = {}
    # Check SAVE/RESTORE mode
    if FLAGS.dict_mode == 'SAVE':
        # Do the predictions for given 'FLAGS.model_dir'/'FLAGS.dataset_name' dataset stored as dic,
        #   and save the dictionary file to 'FLAGS.model_dir' as 'FLAGS.dataset_name'.pickle
        dic = run_predictions(os.path.join(FLAGS.model_dir, FLAGS.dataset_name))
        
        with open(os.path.join(FLAGS.model_dir, FLAGS.dataset_name+'.pickle'), 'wb') as f:
            pickle.dump(dic, f)
        
        print("Saved dictionary to %s" % (os.path.join(FLAGS.model_dir, FLAGS.dataset_name+'.pickle')))
    else:
        # Restore dictionay file from 'FLAGS.model_dir' as dic
        with open(os.path.join(FLAGS.model_dir, FLAGS.dataset_name+'.pickle'), 'rb') as f:
            dic = pickle.load(f)
        print("Restored dictionary from %s" % (os.path.join(FLAGS.model_dir, FLAGS.dataset_name+'.pickle')))

    # Sort the dic with 'FLAGS.sort_method' as sorted_dict:
    #   1. sort the predictions, and only store the top 'FLAGS.num_top_p_predictions'.
    #   2. sort the files according to its indecisiveness and only store the top 'FLAGS.num_top_i_images'.
    sorted_dict = sort_dictionary(dic, method=FLAGS.sort_method)
    
    # Copy top 'FLAGS.num_top_i_images' files from 'FLAGS.model_dir'/'FLAGS.dataset_name' to 'FLAGS.output_dir'/'FLAGS.dataset_name'/indecisive.
    filter_copy_files_to_dir(sorted_dict,
                             os.path.join(FLAGS.output_dir, FLAGS.dataset_name, 'indecisive'), 0)

    # Write sorted_dict to 'FLAGS.output_dir'/'FLAGS.dataset_name'/indecisive/dict+'_'+'FLAGS.num_top_i_images'.pickle
    write_to_file(sorted_dict, 
                  os.path.join(FLAGS.output_dir, FLAGS.dataset_name, 'indecisive'),
                  'dict_' + str(FLAGS.num_top_i_images))
    
    # Check if provide target image
    if FLAGS.target_dir != '':
        # Check if file exist
        if os.path.exists(FLAGS.target_dir):
            # Do prediction for given image and return a dictionary as tar_dic.
            tar_dic = run_predictions(os.path.join(FLAGS.target_dir))
            # Calculate similarities between tar_dic of the images in the 'FLAGS.target_dir' and every distribution in dic and store as sim_dic.
            sim_dic = {}
            # item: [('fox', score),...]
            for tar_key, tar_item in tar_dic.iteritems():
                sim_list = []
                for key, item in dic.iteritems():
                    sim_list.append((key, calculate_log_similarity(tar_item, item)))
                sim_dic[tar_key] = sim_list

            # Sort sim_dic and only store 'FLAGS.num_top_s_similar' as sorted_tar_similarities
            sorted_sim_dic = sort_similarity_dictionary(sim_dic)
            print("Finished calculating similarities")

            # Copy top 'FLAGS.num_top_s_similar' files from 'FLAGS.model_dir'/'FLAGS.dataset_name' to 'FLAGS.output_dir'/'FLAGS.dataset_name'/similar
            filter_copy_files_to_dir(sorted_sim_dic,
                             os.path.join(FLAGS.output_dir, FLAGS.dataset_name, 'similar'), 1)
            print("Fiished copying files")
            # Write sorted_sim_dic to 'FLAGS.output_dir'/'FLAGS.dataset_name'/similar/similar+'_'+'FLAGS.num_top_s_similar'
            write_to_file(sorted_sim_dic, 
                  os.path.join(FLAGS.output_dir, FLAGS.dataset_name, 'similar'),
                  'similar_top_' + str(FLAGS.num_top_i_images))
            print("Fiished writing files")
        else:
            print('Directory does not exist %s' % (FLAGS.target_dir))
    else:
        print("FLAGS.target_dir is not specified: %s" % (FLAGS.target_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--dict_mode',
        type=str,
        default='SAVE',
        help="""\
        SAVE/RESTORE: the prediction results for given dataset.
            filename: 'given_dataset_foldername' + '.pickle'
            location: model_dir
        """
    )
    # classify_image_graph_def.pb:
    #   Binary representation of the GraphDef protocol buffer.
    # imagenet_synset_to_human_label_map.txt:
    #   Map from synset ID to a human readable string.
    # imagenet_2012_challenge_label_map_proto.pbtxt:
    #   Text representation of a protocol buffer mapping a label to synset ID.
    # {classified_folder_name}_dictionary.pickle
    #   
    parser.add_argument(
        '--model_dir',
        type=str,
        default='/tmp/inception',
        help="""\
        Path to classify_image_graph_def.pb,
        imagenet_synset_to_human_label_map.txt, and
        imagenet_2012_challenge_label_map_proto.pbtxt.
        given_dataset_foldername.pickle\
        """
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='sample_1000_images',
        help='Dataset name for provided images'
    )

    parser.add_argument(
        '--sort_method',
        type=str,
        default='default',
        help='Either default or weighted.'
    )
    parser.add_argument(
        '--num_top_p_predictions',
        type=int,
        default=5
    )
    parser.add_argument(
        '--num_top_i_images',
        type=int,
        default=100
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/tmp/inception/output',
        help='Absolute output file path'
    )
    parser.add_argument(
        '--target_dir',
        type=str,
        default='',
        help='Target image directory we provide to be classified to find the most similar image set.'
    )
    parser.add_argument(
        '--num_top_s_similar',
        type=int,
        default=10,
        help='Top s number of most similar images for given image.'
    )
   
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
