#!/home/jelle/.virtualenvs/face-morphing/bin/python
"""
Keeps reading from stdin to store nn progress
"""
import pickle
import sys

import matplotlib.pyplot as plt


def main():

    r = sys.stdin

    closed = False

    loss_curve = list()
    iterations = 0

    test_results = dict()

    while not closed:
        line = r.readline()

        if line == '':
            closed = True

        else:
            splitted = line.split('loss = ')

            # check input
            if len(splitted) >= 2:   # normal output input
                loss_curve.append(float(splitted[1]))
                iterations = iterations + 1

                if iterations % 100 == 0:   # print progress
                    print('Iteration: {}, loss-function: {}'.format(str(iterations), splitted[1]))

            elif 'ITERATION-TEST:' in line:     # intermediate test input
                # get test results
                iters = int(line.split(':')[1])
                tp = int(r.readline().split('tp=')[1])
                fp = int(r.readline().split('fp=')[1])
                tn = int(r.readline().split('tn=')[1])
                fn = int(r.readline().split('fn=')[1])

                APCER = float(r.readline().split('APCER=')[1])
                BPCER = float(r.readline().split('BPCER=')[1])

                test_results[iters] = (tp, fp, tn, fn, APCER, BPCER)

            else:   # unrecognized input
                print('???')
                print(line)


    # save intermediate test results
    f = open('intermediate_test_results.pk', 'wb')
    pickle.dump(test_results, f)


    # plot results in a graph
    INTERVAL = 100
    to_plot = loss_curve[::INTERVAL]
    x = range(1, len(loss_curve) + 1, INTERVAL)
    plt.plot(x, to_plot)
    plt.title('Loss Curve over Iterations')
    plt.ylabel('loss function')
    plt.ylim(0, 1)
    plt.xlabel('iterations')
    plt.savefig('loss-graph.png')


if __name__ == '__main__':
    main()
