#!/home/jelle/.virtualenvs/face-morphing/bin/python
import pickle
import random

from sklearn import svm
from scipy.stats import entropy
import numpy as np
import matplotlib.pyplot as plt

import utils


KERNEL = 'poly'


def main():
    test(combined_det_plot=True)
    plot_kb_det_curves()


def plot_kb_det_curves():
    det_curves = dict()

    for prefix in utils.region_prefixes():
        (fpr, fnr, thresholds) = kullback_leibler_det2(prefix)
        auc = area_under_curve(fpr, fnr)
        det_curves[prefix] = (fpr, fnr, auc)

    # plot results
    plt.figure()
    plt.style.use('ggplot')
    plt.title('KB-DET')
    plt.ylim(0, 1)
    plt.plot([0, 1], [1, 0], linestyle='--')

    for prefix in det_curves:
        PREFIX = prefix
        if prefix == 'RB':
            PREFIX = 'LB'
        if prefix == 'LB':
            PREFIX = 'RB'
        if prefix == 'RC':
            PREFIX = 'LC'
        if prefix == 'LC':
            PREFIX = 'RC'

        (fpr, fnr, auc) = det_curves[PREFIX]
        plt.plot(fnr, fpr, label='{} {:.2f}'.format(PREFIX, auc))

    plt.xlabel('APCER')
    plt.ylabel('BPCER')
    plt.legend(loc='upper right')
    plt.savefig('DET-KB-REGIONS-PLOT.png')
    plt.clf()


def kullback_leibler_det2(prefix):
    data = utils.get_train_test_spectral_data()

    training = data['training']
    test = data['test']

    combined_data = training + test

    (fpr, fnr, thresholds) = kullback_leibler_det2_curve(combined_data, prefix)

    return (fpr, fnr, thresholds)


def kullback_leibler_det2_curve(data, prefix):
    X = list()
    Y = list()

    for pair in data:
        bona_fide = pair[1]
        region_pair = pair[0][prefix]
        divergence = entropy(normalizeL1(region_pair[0]), normalizeL1(region_pair[1]))
        X.append(divergence)

        if bona_fide:
            Y.append(0)
        else:
            Y.append(1)

    (fpr, fnr, thresholds) = det_curve(Y, X)
    return (fpr, fnr, thresholds)







def roc_kullback_leibler(prefix):
    # plot roc graph for divergence approach and threshold
    data = utils.get_train_test_spectral_data()

    training = data['training']
    test = data['test']

    X = list()
    Y = list()

    for pair in training:
        bona_fide = pair[1]
        region_pair = pair[0][prefix]
        divergence = [entropy(normalizeL1(region_pair[0]), normalizeL1(region_pair[1]))]

        X.append(divergence)

        if bona_fide:
            Y.append(0)
        else:
            Y.append(1)

    clf = svm.SVC(kernel=KERNEL, degree=3, tol=1e-3, probability=True, class_weight='balanced')
    clf.fit(X, Y)

    Xt = list()
    Yt = list()

    for pair in test:
        bona_fide = pair[1]
        region_pair = pair[0][prefix]
        divergence = [entropy(normalizeL1(region_pair[0]), normalizeL1(region_pair[1]))]

        Xt.append(divergence)

        if bona_fide:
            Yt.append(0)
        else:
            Yt.append(1)

    scores = clf.predict_proba(X)[:, 1]  # take only positive class probability
    scores_t = clf.predict_proba(Xt)[:, 1]

    fpr, fnr, thresholds = det_curve(Y, scores)
    auc = area_under_curve(fpr, fnr)
    fpr_t, fnr_t, thresholds_t = det_curve(Yt, scores_t)
    auc_t = area_under_curve(fpr_t, fnr_t)

    # Plot results as done in: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
    plt.plot(fpr, fnr, color='darkorange', label="Train set (AUC = {})".format(auc))
    plt.plot(fpr_t, fnr_t, color='darkviolet', label="Test Set (AUC = {})".format(auc_t))
    plt.plot([0, 1], [1, 0], color='navy', linestyle='--')
    plt.xlabel('APCER')
    plt.ylabel('BPCER')
    plt.title('DET-Curve Kullback-Leibler Divergence for {}'.format(prefix))
    plt.legend(loc='upper right')
    plt.savefig('DET-KB-' + prefix + '.png')
    plt.clf()


def det_curve(labels, prob_scores):
    """Create DET Curve
    @param: labels, true class labels {0, 1}
    @param: prob_scores, probability that the sample is positive"""

    probs = sorted(list(set(prob_scores)))

    fpr_list = list()
    fnr_list = list()
    thresholds = list()

    for threshold in probs:

        # assert fpr and tpr
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for i in range(len(labels)):
            morph = labels[i]       # 1 if morp, 0 otherwise
            prob = prob_scores[i]   # probability

            predicted = 0
            if prob > threshold:
                predicted = 1

            if morph == 1 and predicted == 1:
                tp = tp + 1
            elif morph == 1 and predicted == 0:
                fn = fn + 1
            elif morph == 0 and predicted == 1:
                fp = fp + 1
            elif morph == 0 and predicted == 0:
                tn = tn + 1
            else:
                raise Exception('Could not determine confusion class for morph: {}, predicted: {}.'.format(morph, predicted))

        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)

        fpr_list.append(fpr)
        fnr_list.append(fnr)

        thresholds.append(threshold)

    return (fpr_list, fnr_list, thresholds)


def area_under_curve(Y, X):
    """Calculates the area under the curve."""
    auc = 0
    prev = 0

    for i in range(len(X)):
        dx = X[i] - prev
        prev = X[i]
        y = Y[i]
        auc = auc + (y * dx)

    return auc


def kullback_leibler(prefix):

    data = utils.get_train_test_spectral_data()

    training = data['training']

    B = list()
    M = list()

    for pair in training:
        bona_fide = pair[1]
        region_pair = pair[0][prefix]
        divergence = entropy(normalizeL1(region_pair[0]), normalizeL1(region_pair[1]))

        if bona_fide:
            B.append(divergence)
        else:
            M.append(divergence)

    bona_avg = np.average(B)
    bona_var = np.var(B, ddof=1)
    morph_avg = np.average(M)
    morph_var = np.var(M, ddof=1)

    return ((bona_avg, bona_var), (morph_avg, morph_var))


def plot_kullback_leibler(prefix):
    dataset = utils.get_train_test_spectral_data()
    data = dataset['training'] + dataset['test']

    divergence = list()
    random_y_offset = list()
    colors = list()

    for pair in data:
        bona_fide = pair[1]
        region_pair = pair[0][prefix]
        divergence.append(entropy(normalizeL1(region_pair[0]), normalizeL1(region_pair[1])))

        if bona_fide:
            colors.append(0)
        else:
            colors.append(1)

        random_y_offset.append(random.random())

    plt.scatter(divergence, random_y_offset, c=colors, alpha=0.2, cmap='seismic')
    plt.savefig('Kullback-Leibler-Divergence-Plot.png')


def svm_spectral(prefix, pf, class_weight='balanced', probability=False, tol=1e-5, degree=3):
    clf = svm.SVC(kernel=KERNEL, class_weight=class_weight, degree=degree, tol=tol, probability=probability)
    data = utils.get_train_test_spectral_data()

    # convert for use by the svm
    X = list()  # input data
    Y = list()  # class

    training = data['training']

    for pair in training:
        bona_fide = pair[1]
        region_pair = pair[0][prefix]

        vector = pf(region_pair[0], region_pair[1])

        X.append(vector)

        if bona_fide:
            Y.append(0)
        else:
            Y.append(1)

    clf.fit(X, Y)
    return clf


def test_svm_spectral(clf, prefix, pf):
    '''
    param: clf, svm to evaluate
    prefix: prefix for which the svm has been built
    '''

    test_result_dict = {'FP': 0, 'FN': 0, 'TP': 0, 'TN': 0}

    data = utils.get_train_test_spectral_data()
    test = data['test']

    for pair in test:
        bona_fide = pair[1]
        region_pair = pair[0][prefix]

        vector = pf(region_pair[0], region_pair[1])
        vector = [vector]

        prediction = clf.predict(vector)

        if bona_fide and prediction == 1:
            test_result_dict['FP'] = test_result_dict['FP'] + 1
        elif bona_fide and prediction == 0:
            test_result_dict['TN'] = test_result_dict['TN'] + 1
        elif not bona_fide and prediction == 1:
            test_result_dict['TP'] = test_result_dict['TP'] + 1
        elif not bona_fide and prediction == 0:
            test_result_dict['FN'] = test_result_dict['FN'] + 1
        else:
            raise Exception("Could not determine test result of prediction.")

    return test_result_dict


def det_svm_spectral(clf, prefix, pf):
    '''
    @param: clf,    trained classifier with probability prediction enabled.
    @param: prefix, prefix to determine det for
    @param: pf,     preprocessing function for the data
    @returns: result, dictionary of det curve on training and test set.
    '''
    data = utils.get_train_test_spectral_data()
    training = data['training']
    test = data['test']

    X = list()
    Y = list()
    # split up into labels, and prob scores
    for pair in training:
        bona_fide = pair[1]
        region_pair = pair[0][prefix]
        vector = pf(region_pair[0], region_pair[1])
        X.append(vector)
        if bona_fide:
            Y.append(0)     # Negative
        else:
            Y.append(1)     # Positive

    Xt = list()
    Yt = list()
    for pair in test:
        bona_fide = pair[1]
        region_pair = pair[0][prefix]
        vector = pf(region_pair[0], region_pair[1])
        Xt.append(vector)
        if bona_fide:
            Yt.append(0)     # Negative
        else:
            Yt.append(1)     # Positive

    # calculate probability scores
    scores = clf.predict_proba(X)[:, 1]
    scores_t = clf.predict_proba(Xt)[:, 1]

    fpr, fnr, thresholds = det_curve(Y, scores)
    auc = area_under_curve(fpr, fnr)
    fpr_t, fnr_t, thresholds_t = det_curve(Yt, scores_t)
    auc_t = area_under_curve(fpr_t, fnr_t)

    result = dict()
    result['training'] = (fpr, fnr, thresholds, auc)
    result['test'] = (fpr_t, fnr_t, thresholds_t, auc_t)
    return result


def test_svm_frequency(clf, prefix, pf):

    test_result_dict = {'FP': 0, 'FN': 0, 'TP': 0, 'TN': 0}

    data = utils.get_train_test_frequency_data()
    test = data['test']

    for pair in test:
        bona_fide = pair[1]
        region_pair = pair[0][prefix]

        vector = pf(region_pair[0], region_pair[1])
        vector = [vector]

        prediction = clf.predict(vector)

        if bona_fide and prediction == 1:
            test_result_dict['FP'] = test_result_dict['FP'] + 1
        elif bona_fide and prediction == 0:
            test_result_dict['TN'] = test_result_dict['TN'] + 1
        elif not bona_fide and prediction == 1:
            test_result_dict['TP'] = test_result_dict['TP'] + 1
        elif not bona_fide and prediction == 0:
            test_result_dict['FN'] = test_result_dict['FN'] + 1
        else:
            raise Exception("Could not determine test result of prediction.")

    return test_result_dict


def experiment1(vector1, vector2):
    wv1 = weighted_mean(vector1)
    wv2 = weighted_mean(vector2)

    w = wv1[1:] + wv2[1:]
    input_vector = normalizeL1(w)
    return input_vector


def experiment2(vector1, vector2):
    'experiment 1 with log'
    wv1 = weighted_mean(vector1)
    wv2 = weighted_mean(vector2)
    nv1 = np.log(wv1)
    nv2 = np.log(wv2)

    w = nv1[1:] + nv2[1:]
    input_vector = normalizeL1(w)
    return input_vector


def experiment3(vector1, vector2):
    wv1 = weighted_mean(vector1)
    wv2 = weighted_mean(vector2)

    w = wv1[1:] + wv2[1:]
    input_vector = normalizeL2(w)
    return input_vector


def experiment4(vector1, vector2):
    wv1 = weighted_mean(vector1)
    wv2 = weighted_mean(vector2)
    nv1 = np.log(wv1)
    nv2 = np.log(wv2)

    w = nv1[1:] + nv2[1:]
    input_vector = normalizeL2(w)
    return input_vector


def experiment5(vector1, vector2):
    w = vector1[1:] + vector2[1:]
    input_vector = normalizeL1(w)
    return input_vector


def experiment6(vector1, vector2):
    nv1 = np.log(vector1)
    nv2 = np.log(vector2)

    w = nv1[1:] + nv2[1:]
    input_vector = normalizeL1(w)
    return input_vector


def experiment7(vector1, vector2):
    w = vector1[1:] + vector2[1:]
    input_vector = normalizeL2(w)
    return input_vector


def experiment8(vector1, vector2):
    wv1 = np.log(vector1)
    wv2 = np.log(vector2)

    w = wv1[1:] + wv2[1:]
    input_vector = normalizeL2(w)
    return input_vector


def experiment9(vector1, vector2):
    # non-differential
    w = vector2[1:]     # Questioned image
    input_vector = normalizeL1(w)
    return input_vector


def experiment10(vector1, vector2):
    # non-differential
    w = vector2[1:]
    input_vector = normalizeL2(w)
    return input_vector


def experiment11(vector1, vector2):
    # non-differential
    w = vector2[1:]
    nw = np.log(w)

    input_vector = normalizeL1(nw)
    return input_vector


def experiment12(vector1, vector2):
    # non-differential
    w = vector2[1:]
    nw = np.log(w)

    input_vector = normalizeL2(nw)
    return input_vector


def experiment13(vector1, vector2):
    w = np.subtract(vector1[1:],  vector2[1:])
    v = normalizeL2(w)
    return v


def normalizeL1(vector):
    vsum = 0

    for v in vector:
        vsum = vsum + abs(v)

    return np.array(vector) / vsum


def normalizeL2(vector):
    sqsum = 0

    for v in vector:
        sqsum = sqsum + v**2

    sqsum = np.sqrt(sqsum)

    return np.array(vector) / sqsum


def weighted_mean(vector):
    # multiplies mean by the radius of the circle, since there is fewer sample size
    w = np.arange(1, len(vector) + 1)
    return vector * w


def save_experiment_result(path, obj):

    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def plot_combined_det_graph(det_curve_dict, title, filename):

    # plot the results
    plt.figure()
    plt.style.use('ggplot')

    for prefix in det_curve_dict:
        PREFIX = prefix

        if prefix == 'RB':
            PREFIX = 'LB'
        elif prefix == 'LB':
            PREFIX = 'RB'
        elif prefix == 'RC':
            PREFIX = 'LC'
        elif prefix == 'LC':
            PREFIX = 'RC'

        (fpr, fnr, thresholds, auc) = det_curve_dict[PREFIX]
        plt.plot(fnr, fpr, label='{} {:.2f}'.format(PREFIX, auc))

    plt.plot([0, 1], [1, 0], color='navy', linestyle='--')
    plt.xlabel('APCER')
    plt.ylabel('BPCER')
    plt.title(title)
    plt.legend(loc='upper right')
    plt.savefig(filename)
    plt.clf()


def test(plot_det=False, combined_det_plot=False):

    PREPROCESSING_FUNC = experiment12
    EXPERIMENT_NAME = 'experiment12'
    CLASS_WEIGHT = 'balanced'
    DEGREE = 3
    TOL = 1e-8

    result_dict = dict()

    det_curve_dict = dict()

    for prefix in utils.region_prefixes():

        clf = svm_spectral(prefix, PREPROCESSING_FUNC, class_weight=CLASS_WEIGHT, probability=True, tol=TOL, degree=DEGREE)
        test_results = test_svm_spectral(clf, prefix, PREPROCESSING_FUNC)
        result_dict[prefix] = test_results

        if plot_det:

            det_results = det_svm_spectral(clf, prefix, PREPROCESSING_FUNC)

            (fpr, fnr, thresholds, auc) = det_results['training']
            (fpr_t, fnr_t, thresholds_t, auc_t) = det_results['test']

            # plot the results
            plt.plot(fnr, fpr, color='darkorange', label="Train set (AUC = {})".format(auc))
            plt.plot(fnr_t, fpr_t, color='darkviolet', label="Test Set (AUC = {})".format(auc_t))
            plt.plot([0, 1], [1, 0], color='navy', linestyle='--')
            plt.xlabel('APCER')
            plt.ylabel('BPCER')
            plt.title('DET-Curve SVM(' + PREPROCESSING_FUNC.__name__ + ') for {}'.format(prefix))
            plt.legend(loc='upper right')
            plt.savefig('DET_' + PREPROCESSING_FUNC.__name__ + '_' + prefix + '.png')
            plt.clf()

        if combined_det_plot:

            det_results = det_svm_spectral(clf, prefix, PREPROCESSING_FUNC)
            det_curve_dict[prefix] = det_results['test']

    if combined_det_plot:
        plot_combined_det_graph(det_curve_dict, 'DET-SVM-12', '{}-REGIONS-DET.png'.format(PREPROCESSING_FUNC.__name__))

    description = """
    SVM INFO
    Kernel: """ + KERNEL + """,
    Degree: """ + str(DEGREE) + """,
    Tol: """ + str(TOL) + """,
    Class_Weight: """ + str(CLASS_WEIGHT) + """,
    ===================
    DATA INFO
    dataset: 'spectral_raw'
    differential: 'yes'
    preprocessing-function: """ + PREPROCESSING_FUNC.__name__ + """"""

    save_experiment_result(EXPERIMENT_NAME + '.pk', (result_dict, description))


if __name__ == '__main__':
    main()
