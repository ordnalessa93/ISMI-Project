#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
from data_util import create_submission
from generators import PatchSequence
import pandas as pd


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('--model-prefix', type=str)
    args = p.parse_args()

    test_df = pd.read_csv('data/testSetIncludingGrandChallangeName.csv',
                          dtype=str)
    test_seq = PatchSequence('./data', test_df, 32, 'test')
    model_names = ['{}_fold_{}.h5'.format(args.model_prefix, i)
                   for i in range(1, 6)]
    print('Averaging for:')
    print('\n'.join(model_names))
    create_submission(test_df, test_seq, model_names)
    print('Done!')
