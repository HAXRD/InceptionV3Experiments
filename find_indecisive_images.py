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
		

def run_inference_on_images(images_dir):
	"""Runs inference on a list of images.

	Returns:
		A dictionary of top k category-score pairs.
	"""
	if not tf.gfile.IsDirectory(images_dir):
		tf.logging.fatal('Folder does not exist %s', images_dir)
#   image_data = tf.gfile.FastGFile(image, 'rb').read()

	image_top_k_category_score_pair_dictionary = {}

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
		for idx, image_name in enumerate(tf.gfile.ListDirectory(images_dir)):
			image_data = tf.gfile.FastGFile(os.path.join(images_dir, image_name), 'rb').read()
			predictions = sess.run(softmax_tensor,
								  {'DecodeJpeg/contents:0': image_data})
			predictions = np.squeeze(predictions)
		
			# Creates node ID --> English string lookup.
			node_lookup = NodeLookup()

			top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
			category_score_pairs = []
			for node_id in top_k:
				human_string = node_lookup.id_to_string(node_id)
				score = predictions[node_id]
				# print('%s (score = %.5f)' % (human_string, score))
				category_score_pairs.append((human_string,score))
			image_top_k_category_score_pair_dictionary[image_name] = category_score_pairs
			progress_bar(idx + 1, total_length, prefix='Classification process', suffix='Completed.', length=50)
	return image_top_k_category_score_pair_dictionary


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
	if (method == 'weighted'):
		def weighted_lambda(item):
			weighted_sum = 0
			mean = sum([item[1][i][1] for i in range(FLAGS.num_top_predictions)])/FLAGS.num_top_predictions
			for i in range(FLAGS.num_top_predictions):
				curr_score = item[1][i][1]
				weighted_sum = weighted_sum + (curr_score - mean)**2
			return weighted_sum
		return collections.OrderedDict(sorted(given_dictionary.items(), key=weighted_lambda))
	else: # default
		return collections.OrderedDict(sorted(given_dictionary.items(), 
                                    key=lambda it: it[1][0][1], reverse=True))

def write_to_file(dictionary):
	with open(os.path.join(FLAGS.write_dir, 'output.txt'), 'w') as f:
		total_length = len(dictionary)
		for idx, (key, item) in tqdm(enumerate(dictionary.iteritems())):
			f.write('{}. {}\n'.format(idx, key))
			for pair in item:
				f.write('\t%.9f:  %s\n' % (pair[1], pair[0]))
			# progress_bar(idx + 1, total_length, prefix='Writing progress ', suffix='Completed', length=50)

def filter_copy_files_to_dir(given_dictionary):
	if os.path.exists(FLAGS.write_dir):
		shutil.rmtree(FLAGS.write_dir, ignore_errors=True)
	os.makedirs(FLAGS.write_dir)
	
	for key, item in tqdm(enumerate(given_dictionary.iteritems())):
		filename = item[0]
		shutil.copyfile(os.path.join(FLAGS.images_dir, filename), os.path.join(FLAGS.write_dir, str(key)+'.'+filename.split('.')[-1]))


def main(_):
	# image_top_k_category_score_pair_dictionary = collections.OrderedDict(
	# 	sorted(run_inference_on_images(FLAGS.images_dir).items(), key=lambda it: it[1][0][1], reverse=True))
	# for key, item in image_top_k_category_score_pair_dictionary.iteritems():
	# 	print(key)
	# 	for pair in item:
	# 		print('\t%.9f:  %s', (pair[1], pair[0]))
	# write_to_file(image_top_k_category_score_pair_dictionary)
	maybe_download_and_extract()
	sorted_image_top_k_category_score_pair_dictionary = sort_dictionary(
							run_inference_on_images(FLAGS.images_dir), method=FLAGS.sort_method)
	filter_copy_files_to_dir(sorted_image_top_k_category_score_pair_dictionary)
	write_to_file(sorted_image_top_k_category_score_pair_dictionary)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# classify_image_graph_def.pb:
	#   Binary representation of the GraphDef protocol buffer.
	# imagenet_synset_to_human_label_map.txt:
	#   Map from synset ID to a human readable string.
	# imagenet_2012_challenge_label_map_proto.pbtxt:
	#   Text representation of a protocol buffer mapping a label to synset ID.
	parser.add_argument(
        '--model_dir',
        type=str,
        default='/tmp/inception',
        help="""\
        Path to classify_image_graph_def.pb,
        imagenet_synset_to_human_label_map.txt, and
        imagenet_2012_challenge_label_map_proto.pbtxt.\
        """
	)
	parser.add_argument(
        '--images_dir',
        type=str,
        default='/tmp/inception/testing_images',
        help='Absolute folder path to image files.'
	)
	parser.add_argument(
        '--num_top_predictions',
        type=int,
        default=5,
        help='Display this many predictions.'
	)
	parser.add_argument(
		'--sort_method',
		type=str,
		default='default',
		help='Either default or weighted.'
    )
	parser.add_argument(
		'--write_dir',
		type=str,
		default='/tmp/inception/output',
		help='Absolute output file path'
	)
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
