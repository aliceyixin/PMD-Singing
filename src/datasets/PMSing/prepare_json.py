# coding: utf-8
import logging
import json
from pathlib import Path
import random
import os


logger = logging.getLogger(__name__)
logging.basicConfig(filename='prepare.log', level=logging.INFO)

def prepare(dataset_dir, train_json_path, valid_json_path, test_json_path, shuffle_data, computed_dataset_dir):
    # initialization
    singer = dataset_dir.split('/')[-2]
    subset = dataset_dir.split('/')[-3]
    dataset_dir = Path(dataset_dir)
    train_json_path = Path(train_json_path)
    valid_json_path = Path(valid_json_path)
    test_json_path = Path(test_json_path)

    set_names = ['train', 'valid', 'test']
    json_paths = [train_json_path, valid_json_path, test_json_path]
    for path in json_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    # check if this step is already performed, and skip it if so
    skip = True
    for json_path in json_paths:
        if not json_path.exists():
            skip = False
    if skip:
        logger.info('Skip preparation.')
        return

    logger.info('Generate json files.')

    # get all files from json_all and split
    json_name = '{}_{}_corrected.json'.format(subset, singer)
    with open(os.path.join(dataset_dir, json_name), 'r') as f:
        json_all = json.load(f)

    # shuffle dataset option
    def shuffle_dataset(json_all):
        l = list(json_all.items())
        random.shuffle(l)
        return dict(l)

    if shuffle_data ==True:
        json_all = shuffle_dataset(json_all)

    N = len(json_all)
    logger.info('Find %d files to split.' % N)

    # split dataset
    train_data = dict(list(json_all.items())[:int(0.6*N)])
    valid_data = dict(list(json_all.items())[int(0.6*N):int(0.8*N)])
    test_data = dict(list(json_all.items())[int(0.8*N):])
    json_dataset = [train_data, valid_data, test_data]

    N_train = len(train_data)
    N_valid = len(valid_data)
    N_test = len(test_data)
    logger.info('[{}]: {} files in train set, {} files in valid set, {} files in test set.'.format(singer, N_train, N_valid, N_test))
    for json_data, json_path in zip(json_dataset, json_paths):
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=4)