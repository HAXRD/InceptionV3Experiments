import os
import sys
from six.moves import urllib
import tarfile

from config import DIR, CHECKPOINT_FOLDER, DATA_URL

import daiquiri as dqr 
import logging

dqr.setup(level=logging.DEBUG)
logger = dqr.getLogger()

def maybe_download_and_extract():
    """Download and extract model tar file"""
    dest_dir = os.path.join(DIR, CHECKPOINT_FOLDER)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        logger.warning('Creating direcotry {}'.format(dest_dir))
    else:
        logger.warning('Directory already exists {}'.format(dest_dir))
    
    filename = os.path.basename(DATA_URL)
    filepath = os.path.join(dest_dir, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading {0} {1:.1f} %'.format(
                filename, float(count*block_size)/float(total_size)*100.))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        logger.info('Successfully downloaded {0}: {1:.2f} MB'.format(filename, statinfo.st_size/(1024.*1024.)))
    else:
        logger.warning('{} already exist!'.format(filename))
    tarfile.open(filepath, 'r:gz').extractall(dest_dir)

if __name__ == '__main__':
    maybe_download_and_extract()