import os

from download_inceptionV3 import maybe_download_and_extract
from classify_images import get_predictions_dict
from config import DIR

if __name__ == '__main__':
    maybe_download_and_extract()
    dataset_dict = get_predictions_dict(os.path.join(DIR, 'fall11_urls_top30'))
    print('dataset dictionary length: {}'.format(len(dataset_dict)))