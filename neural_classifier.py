#!/home/jelle/.virtualenvs/face-morphing/bin/python
import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import matplotlib.pyplot as plt

import utils
from svm_classification import experiment7, experiment9, experiment10, experiment12, det_curve, area_under_curve

"""
Create a classifier using the keras tensorflow library.
Class Labels: 0 -> bonafide, 1 -> Morph.
"""

DATA = utils.get_train_test_spectral_data()
CLASS_WEIGHT = utils.get_train_spectral_data_class_weights()

# GLOBAL CONSTANTS

EXPERIMENT_NAME = 'experiment10'
TMP_RESULTS_DIR = './tmpresults/'

# architecture
HIDDEN_LAYERS_SHAPE = (8, 8)

# training
EPOCHS = 100
BS = 32
INIT_LR = 0.01
VAL_SPLIT = 0.2

# pre-processing
PF = experiment12


def main():
    test()


def det_results_test():


    for PREFIX in utils.region_prefixes():
        model = create_model_1()
        (model, H) = train_model(model, PF, PREFIX)
        plot_results(model, PF, PREFIX, H)

    return


def plot_results(model, pf, prefix, H):
    PF = pf
    PREFIX = prefix

    det = get_DET_curves(model, PF, PREFIX)
    plot_nn_history(H, 'Training Curve: ' + PREFIX, TMP_RESULTS_DIR+PREFIX + '-training-curve-' + EXPERIMENT_NAME + '.png')
    plot_det_results(det['training'], 'DET-Curve ' + EXPERIMENT_NAME + ' ' + PREFIX, TMP_RESULTS_DIR+'det-'+EXPERIMENT_NAME+'-'+PREFIX+'-plot.png', det_curve_test=det['test'])

    return


def plot_combined_det_curves(det_dict, title, filename):
    plt.figure()
    plt.title(title)
    plt.style.use('ggplot')
    plt.plot([0, 1], [1, 0], linestyle='--', color='navy')

    for prefix in det_dict:
        PREFIX = prefix

        if prefix == 'RB':
            PREFIX = 'LB'
        elif prefix == 'LB':
            PREFIX = 'RB'
        elif prefix == 'RC':
            PREFIX = 'LC'
        elif prefix == 'LC':
            PREFIX = 'RC'

        (fpr, fnr, thresholds, auc) = det_dict[PREFIX]['test']
        plt.plot(fnr, fpr, label='{} {:.2f}'.format(PREFIX, auc))

    plt.xlabel('APCER')
    plt.ylabel('BPCER')
    plt.ylim(0, 1)
    plt.legend(loc='upper right')
    plt.savefig(filename)
    plt.clf()


def test():

    nn_parameters = {'epochs': EPOCHS, 'batch_size': BS, 'initial_learning_rate': INIT_LR, 'validation_split': VAL_SPLIT,
                     'hidden_layers_shape': HIDDEN_LAYERS_SHAPE, 'pf': PF.__name__}

    confusion_matrices = dict()
    det_curves = dict()

    for prefix in utils.region_prefixes():
        PREFIX = prefix
        model = create_model_1()
        (model, H) = train_model(model, PF, PREFIX)

        cfm = compute_confusion_matrix(model, PF, PREFIX)
        confusion_matrices[PREFIX] = cfm

        plot_results(model, PF, PREFIX, H)

        det = get_DET_curves(model, PF, PREFIX)
        det_curves[prefix] = det

    plot_combined_det_curves(det_curves, 'DET-{}'.format(EXPERIMENT_NAME), 'DET-{}-PLOT.png'.format(EXPERIMENT_NAME))


    # Save results
    DESCRIPTION = '''\n\
            EXPERIMENT INFO:\n\
            experiment_name: {experiment_name}\n\
            \n\
            NEURAL NETWORK INFO\n\
            epochs: {epochs}\n\
            batch_size: {batch_size}\n\
            initial_learning_rate: {initial_learning_rate}\n\
            validation_split: {validation_split}\n\
            hidden_layers_shape: {hidden_layers_shape}\n
            preprocessing_function: {pf}\n'''.format(experiment_name=EXPERIMENT_NAME, **nn_parameters)

    with open(EXPERIMENT_NAME + '.pk', 'wb') as f:
        pickle.dump({'description': DESCRIPTION, 'cfms': confusion_matrices}, f)

    return


def get_DET_curves(model, pf, prefix):
    '''
    Creates DET-curve for both training and test set.
    @param: model,  Trained model capable of producing probabilities.
    @param: pf,     Preprocessing function to use.
    @param: prefix, Prefix that model has been trained on.
    '''

    # create labels (0 and 1)
    # create probability scores (of positive label)

    # train data
    training_data = np.array(DATA['training'])

    Y = list(map(bona_fide_to_integer, training_data[:, 1]))

    X = list()
    for pair in training_data:
        region_pair = pair[0][prefix]
        v = pf(region_pair[0], region_pair[1])
        X.append(v)
    X = np.array(X)     # convert into numpy array

    probs = model.predict(X)
    probs_positive = probs[:, 1]

    fpr, fnr, thresholds = det_curve(Y, probs_positive)
    auc = area_under_curve(fpr, fnr)

    # test data

    test_data = np.array(DATA['test'])

    Y_test = list(map(bona_fide_to_integer, test_data[:, 1]))

    X_test = list()
    for pair in test_data:
        region_pair = pair[0][prefix]
        v = pf(region_pair[0], region_pair[1])
        X_test.append(v)
    X_test = np.array(X_test)     # convert into numpy array

    probs_test = model.predict(X_test)
    probs_positive_test = probs_test[:, 1]

    fpr_test, fnr_test, thresholds_test = det_curve(Y_test, probs_positive_test)
    auc_test = area_under_curve(fpr_test, fnr_test)

    return {'training': (fpr, fnr, thresholds, auc), 'test': (fpr_test, fnr_test, thresholds_test, auc_test)}


def plot_det_results(det_curve_training, title, filename, det_curve_test=None):
    '''
    Plot the det curve
    '''
    plt.figure()
    plt.style.use('seaborn')
    plt.title(title)
    plt.ylim(0, 1)
    plt.plot(det_curve_training[1], det_curve_training[0], label='training set', color='navy')

    if det_curve_test is not None:
        plt.plot(det_curve_test[1], det_curve_test[0], label='test set', color='darkorange')

    plt.plot([0, 1], [1, 0], 'b--')

    plt.xlabel('APCER')
    plt.ylabel('BPCER')
    plt.legend(loc='upper right')
    plt.savefig(filename)
    plt.clf()
    return


def bona_fide_to_integer(bona_fide):
    if bona_fide:
        return 0
    else:
        return 1


def compute_confusion_matrix(model, pf, prefix, data=DATA['test']):
    '''
    Computes confusion matrix. Expects that the model outputs logits.
    With second indexed class being the morph (True).
    Model needs to output probabilities: [bona-fide, morph], with sum = 1.
    '''
    # prep data
    X = list()
    Y = list()

    for pair in data:

        bona_fide = pair[1]
        v = pf(pair[0][prefix][0], pair[0][prefix][1])
        X.append(v)

        if bona_fide:
            Y.append(0)
        else:
            Y.append(1)

    X = np.array(X)
    Y = np.array(Y)

    predictions = model.predict(X)

    cfm = {'FP': 0, 'FN': 0, 'TP': 0, 'TN': 0}

    for i in range(len(X)):
        # create matrix
        pred = predictions[i]
        pred_class = tf.argmax(pred)
        y_true = Y[i]

        if pred_class == 0 and y_true == 0:
            cfm['TN'] = cfm['TN'] + 1

        elif pred_class == 0 and y_true == 1:
            cfm['FN'] = cfm['FN'] + 1

        elif pred_class == 1 and y_true == 0:
            cfm['FP'] = cfm['FP'] + 1

        elif pred_class == 1 and y_true == 1:
            cfm['TP'] = cfm['TP'] + 1

        else:
            raise Exception("Could not determine confusion matrix entry")

    return cfm


def create_model_1():
    # create model architecture
    model = keras.Sequential()

    for layer in HIDDEN_LAYERS_SHAPE:
        model.add(layers.Dense(layer, activation='sigmoid'))
    print(model)

    model.add(layers.Dense(2, activation='softmax'))  # output layer

    loss = keras.losses.BinaryCrossentropy()
    optimizer = keras.optimizers.Adam(learning_rate=INIT_LR)

    metrics = ['binary_accuracy']

    # compile model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics)

    return model


def train_model(model, pf, prefix):
    '''
    @param: model,  keras model to use.
    @param: pf,     preprocessing function.
    @param: prefix, region prefix to train on.
    '''

    training_data = DATA['training']

    (X, Y) = get_data_and_labels(training_data, pf, prefix)

    H = model.fit(X, Y,
             class_weight=CLASS_WEIGHT,
             epochs=EPOCHS,
             validation_split=VAL_SPLIT)

    return (model, H)


def evaluate_model(data, model, pf, prefix):

    (X, Y) = get_data_and_labels(data, pf, prefix)

    return model.evaluate(X, Y)


def get_data_and_labels(data, pf, prefix):
    """
    Morph:      [0, 1]
    Bona-Fide:  [1, 0]
    """

    X = list()
    Y = list()

    for pair in data:
        bona_fide = pair[1]
        region_pair = pair[0][prefix]

        v = pf(region_pair[0], region_pair[1])
        X.append(v)

        if bona_fide:
            Y.append([1, 0])
        else:
            Y.append([0, 1])

    return (np.array(X), np.array(Y))



def plot_nn_history(H, title, filename):
    N = np.arange(0, EPOCHS)

    plt.style.use('seaborn')
    plt.figure()
    plt.plot(N, H.history['loss'], label='train_loss')
    plt.plot(N, H.history['val_loss'], label='val_loss')
    plt.plot(N, H.history['binary_accuracy'], label='train_accuracy')
    plt.plot(N, H.history['val_binary_accuracy'], label='val_accuracy')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss/Accuracy')
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(filename)
    plt.clf()




def get_test_pair(prefix):
    test_data = DATA['test']
    pair = test_data[3]

    bona_fide = pair[1]
    region_pair = pair[0][prefix]

    return (region_pair, bona_fide)


if __name__ == '__main__':
    main()
