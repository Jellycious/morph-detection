#!/home/jelle/.virtualenvs/face-morphing/bin/python
# ENSURES THE SCRIPT IS RUN WITHIN VIRTUAL ENVIRONMENT
import numpy as np

import utils

def main():
    data = utils.get_train_test_spectral_data()
    training = data['training']
    test = data['test']

    highest = 0
    for pair in training:
        pair_dict = pair[0]

        for key in pair_dict:
            trusted = pair_dict[key][0]
            questioned = pair_dict[key][1]
            vect = trusted + questioned
            maxn = np.amax(vect)
            highest = max(highest, maxn)

    for pair in test:
        pair_dict = pair[0]

        for key in pair_dict:
            trusted = pair_dict[key][0]
            questioned = pair_dict[key][1]
            vect = trusted + questioned
            maxn = np.amax(vect)
            highest = max(highest, maxn)

    print(type(highest))
    return

if __name__ == '__main__':
    main()

