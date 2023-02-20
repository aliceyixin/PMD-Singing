# coding: utf-8
import logging
import json
from pathlib import Path
import random
import os
from itertools import islice


logger = logging.getLogger(__name__)
logging.basicConfig(filename='prepare_da_json.log', level=logging.INFO)

def prepare(source_dataset_dir, target_dataset_dir, train_json_path, valid_json_path, test_json_path, shuffle_data, computed_dataset_dir, N_data):
    # initialization
    source_singer = source_dataset_dir.split('/')[-2]
    target_singer = target_dataset_dir.split('/')[-2]
    source_subset = source_dataset_dir.split('/')[-3]
    target_subset = target_dataset_dir.split('/')[-3]

    source_dataset_dir = Path(source_dataset_dir)
    target_dataset_dir = Path(target_dataset_dir)
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

    # get all files from source and target json file and split
    source_json_name = '{}_{}_corrected.json'.format(source_subset, source_singer)
    with open(os.path.join(source_dataset_dir, source_json_name), 'r') as f:
        source_json_all = json.load(f)
    target_json_name = '{}_{}_corrected.json'.format(target_subset, target_singer)
    with open(os.path.join(target_dataset_dir, target_json_name), 'r') as f:
        target_json_all = json.load(f)

    # shuffle dataset option
    def shuffle_dataset(json_all):
        l = list(json_all.items())
        random.shuffle(l)
        return dict(l)

    if shuffle_data ==True:
        source_json_all = shuffle_dataset(source_json_all)
        target_json_all = shuffle_dataset(target_json_all)

    Ns = len(source_json_all)
    Nt = len(target_json_all)
    logger.info('Find {} files in source and {} files in target to split.'.format(Ns, Nt))

    source_list_all = list(source_json_all.items())
    target_list_all = list(target_json_all.items())

    source_list_train = dict(source_list_all[:int(0.6*Ns)])
    source_list_valid = dict(source_list_all[int(0.6*Ns):int(0.8*Ns)])
    source_list_test = dict(source_list_all[int(0.8*Ns):])
    target_list_train = dict(target_list_all[:int(0.6*Nt)])
    target_list_valid = dict(target_list_all[int(0.6*Nt):int(0.8*Nt)])
    target_list_test = dict(target_list_all[int(0.8*Nt):])

    # Required dataset size must larger than true size
    assert N_data >= max(Ns, Nt)

    def take(n, iterable):
        "Return first n items of the iterable as a list"
        return list(islice(iterable, n))   

    def repeat_dataset(data, N_data, percentage):
        repeated_data = {}
        times = (int(percentage * N_data) // len(data)) + 1
        for k in range(times):
            for item in data:
                repeat_id = item + 'k' + str(k) 
                repeated_data[repeat_id] = data[item]
        if percentage == 0.6:
            repeated_data = take(int(percentage * N_data)+1, repeated_data.items())
        else:
            repeated_data = take(int(percentage * N_data), repeated_data.items())

        return repeated_data

    source_train = repeat_dataset(source_list_train, N_data, 0.6)
    source_valid = repeat_dataset(source_list_valid, N_data, 0.2)
    source_test = repeat_dataset(source_list_test, N_data, 0.2)

    target_train = repeat_dataset(target_list_train, N_data, 0.6)
    target_valid = repeat_dataset(target_list_valid, N_data, 0.2)
    target_test = repeat_dataset(target_list_test, N_data, 0.2)

    assert (len(source_train) + len(source_valid)) == (len(target_train) + len(target_valid))

    ## take turns to put source and target to a list
    def iterate_STdata(source, target):
        data = {}
        for source_item, target_item in zip(source, target):
            data[source_item[0]] = source_item[1]
            data[target_item[0]] = target_item[1]
        return data

    train_data = iterate_STdata(source_train, target_train)
    valid_data = iterate_STdata(source_valid, target_valid)
    test_data = iterate_STdata(source_test, target_test)
    json_dataset = [dict(train_data), dict(valid_data), dict(test_data)]

    N_train = len(train_data)
    N_valid = len(valid_data)
    N_test = len(test_data)
    logger.info('[{}2{}]: {} files in train set, {} files in valid set, {} files in test set.'.format(source_singer, target_singer, N_train, N_valid, N_test))
    print('[{}2{}]: {} files in train set, {} files in valid set, {} files in test set.'.format(source_singer, target_singer, N_train, N_valid, N_test))
    for json_data, json_path in zip(json_dataset, json_paths):
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=4)
