#!/home/jelle/.virtualenvs/face-morphing/bin/python
import pickle
import os

from sklearn.neural_network import MLPClassifier
from svm_classification import normalizeL2
import numpy as np

import utils

FREQ_DATA = utils.get_train_test_frequency_data_balanced()
SVM_DATA = utils.get_train_test_spectral_data()


def main():
    # setup pipe to get learning progress hehehe
    test()


def experiment1(trusted, questioned):
    magn_t = np.abs(trusted)[int(len(trusted) / 2), :]     # half of freq can be ignored due to symmetry
    magn_q = np.abs(questioned)[int(len(questioned) / 2), :]

    flat_t = magn_t.flatten()
    flat_q = magn_q.flatten()

    vector = flat_t - flat_q

    vector = normalizeL2(vector)
    return vector


def experiment2(trusted, questioned):
    magn_t = np.abs(trusted)[int(len(trusted) / 2), :]     # half of freq can be ignored due to symmetry
    magn_q = np.abs(questioned)[int(len(questioned) / 2), :]

    flat_t = magn_t.flatten()
    flat_q = magn_q.flatten()

    vector = flat_t + flat_q
    # take the log without negatives

    vector = normalizeL2(log_vector)
    return vector


def get_mlp(parameters):

    clf = MLPClassifier(**parameters, verbose=10)
    return clf


def create_train_nn(prefix, pf, parameters, intermediate_test_interval=None):
    '''
    @param: prefix,         region prefix to train for.
    @param: pf,             pre-processing function.
    @param: parameters,     dictionary of neural_network parameters
    @param: test_interval,  iteration interval to do tests between
    clf = get_mlp()
    '''
    clf = get_mlp(parameters)

    X = list()
    Y = list()

    for pair in DATA['training']:
        bona_fide = pair[1]
        region_pair = pair[0][prefix]
        input_vector = pf(region_pair[0], region_pair[1])
        X.append(input_vector)

        if bona_fide:
            Y.append(0)
        else:
            Y.append(1)


    if intermediate_test_interval:

        iterations = 0

        while iterations < parameters['max_iter']:
            for i in range(intermediate_test_interval):
                clf = clf.partial_fit(X, Y, [0, 1])

            iterations = iterations + intermediate_test_interval
            test = test_classifier(clf, prefix, pf)
            # send results to stdout
            print('ITERATION-TEST:{}'.format(iterations))
            for key in test:
                print(key+'='+str(test[key]))






    else:
        clf.fit(X, Y)

    return clf

def test_classifier(clf, prefix, pf):
    '''Test classifier trained for specific prefix'''

    Xt = list()
    Yt = list()

    for pair in DATA['test']:
        bona_fide = pair[1]
        region_pair = pair[0][prefix]
        input_vector = pf(region_pair[0], region_pair[1])
        Xt.append(input_vector)

        if bona_fide:
            Yt.append(0)
        else:
            Yt.append(1)

    y_predicted = clf.predict(Xt)

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in range(len(Yt)):
        bona_fide = Yt[i]
        y = y_predicted[i]
        if bona_fide == 0 and y == 0:
            tn = tn + 1
        elif bona_fide == 0 and y == 1:
            fp = fp + 1
        elif bona_fide == 1 and y == 0:
            fn = fn + 1
        elif bona_fide == 1 and y == 1:
            tp = tp + 1
        else:
            raise Exception("Could not determine entry for confusion matrix, bona_fide: {}, y_predicted: {}.".format(bona_fide, y_predicted))

    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    APCER = fnr
    BPCER = fpr

    return {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 'APCER': APCER, 'BPCER': BPCER}


def test():

    PREPROCESSING_FUNCTION = experiment1
    ACTIVATION = 'logistic'
    HIDDEN_LAYER_SIZE = (600, 600)
    SOLVER = 'adam'
    ALPHA = 0.001   # L2 penalty (regularization)
    LEARNING_RATE = 'invscaling'
    LEARNING_RATE_INIT = 0.01
    MAX_ITER = 20000
    RANDOM_STATE = 2
    TOL = 1e-5
    N_ITER_NO_CHANGE = 2000

    parameters = {'hidden_layer_sizes': HIDDEN_LAYER_SIZE, 'activation': ACTIVATION, 'solver': SOLVER,
         'alpha': ALPHA, 'learning_rate': LEARNING_RATE, 'learning_rate_init': LEARNING_RATE_INIT, 'max_iter': MAX_ITER, 'random_state': RANDOM_STATE,
         'tol': TOL, 'n_iter_no_change': N_ITER_NO_CHANGE}

    DESCRIPTION = '''\n\
            MLP PARAMETERS\n\
            activation: {activation},\n\
            hidden_layers: {hidden_layer_sizes},\n\
            solver: {solver},\n\
            alpha: {alpha},\n\
            learning_rate: {learning_rate},\n\
            learning_rate_init: {learning_rate_init},\n\
            max_iter: {max_iter},\n\
            random_state: {random_state},\n\
            tol: {tol},\n\
            n_iter_no_change: {n_iter_no_change},\n\
            \n\
            PREPROCESSING: {preprocessing}\n'''.format(preprocessing=PREPROCESSING_FUNCTION, **parameters)

    test = dict()

    #for prefix in utils.region_prefixes():
    #    clf = create_train_nn(prefix, PREPROCESSING_FUNCTION, parameters)
    #    test_result = test_classifier(clf, prefix, PREPROCESSING_FUNCTION)
    #    test[prefix] = (test_result, clf)
    PREFIX = 'NO'
    INTERMEDIATE_TEST_INTERVAL = 100
    clf = create_train_nn(PREFIX, PREPROCESSING_FUNCTION, parameters, intermediate_test_interval=INTERMEDIATE_TEST_INTERVAL)
    test_result = test_classifier(clf, PREFIX, PREPROCESSING_FUNCTION)
    test[PREFIX] = (test_result, clf)

    save_nn_training('mlp-1-' + PREFIX + '.pk', test, DESCRIPTION)


def save_nn_training(path, result_dict, description):
    with open(path, 'wb') as f:
        pickle.dump((result_dict, description), f)


if __name__ == '__main__':
    main()
