#!/home/jelle/.virtualenvs/face-morphing/bin/python
'''
Processes the raw images into alternative presentations neccesary for the research
'''
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np

PAIR_DIR = './datasets/prepared_data/pairs'
TRAINING_DATA_DIR = './datasets/training_data'
TEST_PAIR_DIR = 'B_FRGC_04201'


def pairs_iter():
    '''
    Iterator for the pairs in the dataset.
    Every entry is represented as a tuple of (T, Q).
    '''

    for pair_dir in os.listdir(PAIR_DIR):

        pair_path = os.path.join(PAIR_DIR, pair_dir)

        trusted = None
        questioned = None

        for filename in os.listdir(pair_path):
            if filename[0] == 'Q':
                questioned = os.path.abspath(os.path.join(pair_path, filename))
            elif filename[0] == 'T':
                trusted = os.path.abspath(os.path.join(pair_path, filename))
            else:
                continue

        yield (trusted, questioned)


def get_fpr(cfm):
    # AKA BPCER
    return cfm['FP'] / (cfm['FP'] + cfm['TN'])


def get_fnr(cfm):
    # AKA APCER
    return cfm['FN'] / (cfm['TP'] + cfm['FN'])


def get_tpr(cfm):
    # AKA Sensitivity, Recall
    return cfm['TP'] / (cfm['TP'] + cfm['FN'])


def get_tnr(cfm):
    # AKA Specificity, Selectivity
    return cfm['TN'] / (cfm['FP'] + cfm['TN'])


def get_accuracy(cfm):
    return (cfm['TP'] + cfm['TN']) / (cfm['TP'] + cfm['TN'] + cfm['FP'] + cfm['FN'])


def get_region(dirname, region_prefix):
    '''
    Returns the region of the trusted and questioned image.

    :param dirname: directory of pair
    :param region_prefix: prefix of the region.
    '''
    trusted, questioned = None, None

    for filename in os.listdir(dirname):

        if filename[0:len(region_prefix)] == region_prefix:
            if filename[len(region_prefix) + 1] == 'Q':
                questioned = os.path.abspath(os.path.join(dirname, filename))
            elif filename[len(region_prefix) + 1] == 'T':
                trusted = os.path.abspath(os.path.join(dirname, filename))
            else:
                raise Exception("Couldn't identify potential regionfile: {}".format(filename))

    if trusted is None or questioned is None:
        raise Exception("Could not find trusted or/and questioned region file in dir {}".format(dirname))
        return

    return (trusted, questioned)


def get_regions(dirname):
    '''
    Returns all regions in the form of [(prefix, trusted, questioned)]
    '''
    result = list()
    REGION_PREFIXES = ['CH', 'FH', 'LB', 'LC', 'NO', 'RB', 'RC']
    for prefix in REGION_PREFIXES:
        (trusted, questioned) = get_region(dirname, prefix)
        result.append((prefix, trusted, questioned))

    return result


def region_prefixes():
    return ['CH', 'FH', 'LB', 'LC', 'NO', 'RB', 'RC']


def get_test_image(region=True):
    if region:
        return get_region(os.path.join(PAIR_DIR, TEST_PAIR_DIR), 'CH')[0]
    else:
        return next(pairs_iter())[0]


def get_train_test_spectral_data(log=False):
    '''
    Returns the training and test set of spectral data on the regions
    returns: {'training': training_set,'test': test_set)
    Both with the following structure:
    {region_prefix: (trusted_data, questioned_data)}, bona_fide)
    '''
    if log:
        with open(os.path.join(TRAINING_DATA_DIR, 'train_test_spectral_data.pk'), 'rb') as f:
            data = pickle.load(f)
    else:
        with open(os.path.join(TRAINING_DATA_DIR, 'train_test_spectral_data_RAW.pk'), 'rb') as f:
            data = pickle.load(f)

    return data


def get_train_spectral_data_class_weights():
    '''
    NOTE! Only use class weights with get_train_test_spectral_data. and not with the balanced version.
    '''
    mc = 0
    bfc = 0
    with open(os.path.join(TRAINING_DATA_DIR, 'train_test_spectral_data_RAW.pk'), 'rb') as f:
        data = pickle.load(f)
        training = data['training']

        for pair in training:
            bona_fide = pair[1]

            if bona_fide:
                bfc = bfc + 1
            else:
                mc = mc + 1

    total = mc + bfc
    bf_weight = ((1 / bfc) * (total)) / 2
    mc_weight = ((1 / mc) * (total)) / 2

    return {0: bf_weight, 1: mc_weight}




def get_train_test_spectral_data_balanced():
    with open(os.path.join(TRAINING_DATA_DIR, 'train_test_spectral_data_RAW.pk'), 'rb') as f:
        data = pickle.load(f)

        morph_pairs = list()
        bona_fide_pairs = list()

        for pair in data['training']:
            bona_fide = pair[1]
            if bona_fide:
                bona_fide_pairs.append(pair)
            else:
                morph_pairs.append(pair)

        bfc = len(bona_fide_pairs)
        morph_selection = random.sample(morph_pairs, bfc)
        new_training_data = morph_selection + bona_fide_pairs
        random.shuffle(new_training_data)

        result = dict()
        result['training'] = new_training_data
        result['test'] = data['test']

        return result




def get_train_test_frequency_data(log=False):
    '''
    Returns the training and test set of spectral data on the regions
    returns: (training_set, test_set)
    ({region_prefix: (trusted_data, questioned_data)}, bona_fide, pair_name)
    '''
    if log:
        with open(os.path.join(TRAINING_DATA_DIR, 'train_test_frequency_data.pk'), 'rb') as f:
            data = pickle.load(f)
    else:
        with open(os.path.join(TRAINING_DATA_DIR, 'train_test_frequency_data_RAW.pk'), 'rb') as f:
            data = pickle.load(f)

    return data


def get_train_test_frequency_data_balanced():
    with open(os.path.join(TRAINING_DATA_DIR, 'train_test_frequency_data_RAW.pk'), 'rb') as f:
        data = pickle.load(f)

        morph_pairs = list()
        bona_fide_pairs = list()
        for pair in data['training']:
            bona_fide = pair[1]
            if bona_fide:
                bona_fide_pairs.append(pair)
            else:
                morph_pairs.append(pair)

        # there will be less bona_fide than morphs
        bfc = len(bona_fide_pairs)
        morph_selection = random.sample(morph_pairs, bfc)
        new_training_data = morph_selection + bona_fide_pairs
        random.shuffle(new_training_data)

        result = dict()
        result['training'] = new_training_data
        result['test'] = data['test']

        return result


def spectral_plot_example():
    return



if __name__ == '__main__':
    '''
    For testing purposes only
    '''

    vec = get_train_test_spectral_data()['training'][0][0]['CH'][0]
    print(min(vec))
    print(max(vec[1:]))
