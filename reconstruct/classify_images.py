
import os 
import re
import numpy as np 
import tensorflow as tf 
from tqdm import tqdm
import pickle

from config import DIR, CHECKPOINT_FOLDER, DICTIONARY_FOLDER

import daiquiri as dqr 
import logging

dqr.setup(level=logging.DEBUG)
logger = dqr.getLogger()

class NodeLookup(object):
    def __init__(self, label_lookup_path=None, uid_lookup_path=None):
        if not label_lookup_path:
            label_lookup_path = os.path.join(
                DIR, CHECKPOINT_FOLDER, 'imagenet_2012_challenge_label_map_proto.pbtxt')
        if not uid_lookup_path:
            uid_lookup_path = os.path.join(
                DIR, CHECKPOINT_FOLDER, 'imagenet_synset_to_human_label_map.txt')
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

def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(os.path.join(
        DIR, CHECKPOINT_FOLDER, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

def run_predictions(images_dir: str):
    """Runs predictions on a list of images.

    Args:
        image_dir: directory of given image(s).

    Returns:
        A dictionary of top k category-score pairs.
    """
    # Check if directory is valid
    if not tf.gfile.IsDirectory(images_dir):
        tf.logging.fatal('Folder does not exist {}'.format(images_dir))
    
    # {'n001.jpg': [('dog', 0.5), ...], ...}
    predictions_dictionary = {}

    node_lookup = NodeLookup()

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
        
        # Run predictions for a list of images and store them in a dictionary 
        for image_name in tqdm(tf.gfile.ListDirectory(images_dir)):
            image_data = tf.gfile.FastGFile(os.path.join(images_dir, image_name), 'rb').read()
            try:
                predictions = sess.run(
                    softmax_tensor, {'DecodeJpeg/contents:0': image_data})
                predictions = np.squeeze(predictions)

                # Convert Node UID into human-readable string
                readable_predictions = []
                for node_id, score in enumerate(predictions):
                    human_string = node_lookup.id_to_string(node_id)
                    readable_predictions.append((human_string, score))
                predictions_dictionary[image_name] = readable_predictions
            except:
                print('Error occurred!')
    return predictions_dictionary

def get_predictions_dict(images_dir: str):
    """Restore from the dict file or run 'run_predictions' and Save to dict file."""
    dest_folder = os.path.join(DIR, DICTIONARY_FOLDER)
    dest_file = os.path.join(DIR, DICTIONARY_FOLDER, os.path.basename(images_dir)+'.pickle')
    # Check if DICTIONARY_FOLDER exists, makedir if not
    if not os.path.exists(dest_folder):
        logger.info('Creating folder {}.'.format(dest_folder))
        os.makedirs(dest_folder)
    else:
        logger.warning('Folder already exist {}.'.format(dest_folder))
    
    # Check if dict exists
    if os.path.exists(dest_file): # RESTORE
        logger.info('{0} Start RESTORE Dictionary {0}'.format('='*10))
        with open(dest_file, 'rb') as f:
            predictions_dictionary = pickle.load(f)
        logger.info('Restored dictionary from {}'.format(dest_file))
        return predictions_dictionary
    else: # Run predictions and SAVE
        logger.info('{0} Start Run Predictions {0}'.format('='*10))
        predictions_dictionary = run_predictions(images_dir)
        with open(dest_file, 'wb') as f:
            pickle.dump(predictions_dictionary, f)
        logger.info('Saved dictionary to {}'.format(dest_file))
        return predictions_dictionary

if __name__ == '__main__':
    dataset_dict = get_predictions_dict(os.path.join(DIR, 'fall11_urls_top30'))
    print('dataset dictionary length: {}'.format(len(dataset_dict)))