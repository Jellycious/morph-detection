#!/home/jelle/.virtualenvs/face-morphing/bin/python
import matplotlib.pyplot as plt

import svm_classification as svmc
import neural_classifier as nnc

import utils

"""
The goal of this file is to plot the DET curves of both the NN and SVM.
To compare both classification methods.
"""


def main():

    PF = svmc.experiment7

    for PREFIX in utils.region_prefixes():
        det_svm = get_svm_det_curve(PF, PREFIX)
        det_nn = get_nn_det_curve(PF, PREFIX)

        plot_det_curves(det_svm, det_nn, 'Detection Trade-off', 'det-comparison-' + PREFIX + '.png')
    return


def get_svm_det_curve(pf, prefix):
    clf = svmc.svm_spectral(prefix, pf, probability=True)
    det = svmc.det_svm_spectral(clf, prefix, pf)
    return det['test']


def get_nn_det_curve(pf, prefix):
    nn = nnc.create_model_1()
    (nn, H) = nnc.train_model(nn, pf, prefix)
    det = nnc.get_DET_curves(nn, pf, prefix)
    return det['test']


def plot_det_curves(det_curve_svm, det_curve_nn, title, filename):
    (fpr_svm, fnr_svm, thresholds_svm, auc_svm) = det_curve_svm
    (fpr_nn, fnr_nn, thresholds_nn, auc_nn) = det_curve_nn

    plt.figure()
    plt.style.use('classic')
    plt.ylim(0, 1)

    plt.plot(fnr_svm, fpr_svm, color='darkorange', label='SVM (AUC = {:.2f})'.format(auc_svm))
    plt.plot(fnr_nn, fpr_nn, color='darkviolet', label='NN (AUC = {:.2f})'.format(auc_nn))
    plt.plot([0, 1], [1, 0], color='navy', linestyle='--')
    plt.xlabel('APCER')
    plt.ylabel('BPCER')
    plt.title(title)
    plt.legend(loc='upper right')
    plt.savefig(filename)
    plt.clf()
    return


if __name__=='__main__':
    main()
