#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from data_util import load_image, get_patch
from os.path import join
import keras
import keras.backend as K
import numpy as np
import pandas as pd


class PatchGenerator:

    def __init__(self, input_dir, dataframe, batch_size, dataset = 'train', res='v', augmentation_fn=None):
        self.input_dir = input_dir
        self.batch_size = batch_size
        self.dataset = dataset
        self.res = res
        self.augmentation_fn = augmentation_fn
        self.df = dataframe

        self.n_samples = len(self.df)
        self.n_batches = self.n_samples // self.batch_size

        print('PatchGenerator detected: {n_samples} patch samples.'.format(n_samples=self.n_samples))

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self.n_batches

    def next(self):
        df_2 = self.df.loc[self.df['histology'] == '2', :].sample(self.batch_size//3, replace=False)
        df_20 = self.df.loc[self.df['histology'] == '20', :].sample(self.batch_size//3, replace=False)
        rest = self.batch_size - (len(df_2) + len(df_20))
        df_21 = self.df.loc[self.df['histology'] == '21', :].sample(rest, replace=False)

        df_batch = pd.concat([df_2, df_20, df_21])

        images = []
        labels = []
        for index, row in df_batch.iterrows():
            try:
                pID = row['patientID']
                sID = row['studyID']
                sNa = row['scanName']

                image_full = load_image(pID, sID, sNa, datadir=self.input_dir, dataset = self.dataset, res=self.res)
                label = row['histology']

                if self.res == 'v':
                    infix = 'Low'
                else:
                    infix = 'High'
                coordinates = (int(row['annotation{}Resolution{}'.format(infix, i)]) for i in [1, 2, 3])

                image = get_patch(image_full, coordinates, size=(70,70,40))

                if self.augmentation_fn:
                    image = self.augmentation_fn(image)

                images.append(image)

                if label == '2':
                    labels.append((1, 0, 0))
                elif label == '20':
                    labels.append((0, 1, 0))
                elif label == '21':
                    labels.append((0, 0, 1))
            except Exception as e:
                print('Failed reading idx {idx}...'.format(idx=index))

        batch_x = np.stack(images).astype(K.floatx())
        batch_y = np.stack(labels).astype(K.floatx())

        return batch_x, batch_y


class PatchSequence(keras.utils.Sequence):

    def __init__(self, input_dir, dataframe, batch_size, dataset='train', res='v'):
        self.input_dir  = input_dir
        self.df         = dataframe
        self.batch_size = batch_size
        self.res        = res
        self.dataset    = dataset

        self.n_samples = len(self.df)
        self.n_batches = int(np.ceil(self.n_samples / self.batch_size))

        # Print some info
        print('PatchSequence detected: {n_samples} patch samples.'.format(n_samples=len(self.df)))

    def __len__(self):
        return self.n_batches

    def get_all_labels(self):
        if self.dataset == 'train':
            return self.df.loc[:, 'histology'].values.astype(K.floatx())
        return None

    def __getitem__(self, idx):
        # idx indexes batches, not samples

        # Provide batches of samples
        images = []
        labels = []

        # Create indexes for samples
        idx1 = idx * self.batch_size
        idx2 = np.min([idx1 + self.batch_size, self.n_samples])
        idxs = np.arange(idx1, idx2)

        # Iterate over samples
        for i in idxs:
            try:

                # get the row
                row = self.df.iloc[i, :]

                # read data and label
                pID = row['patientID']
                sID = row['studyID']
                sNa = row['scanName']

                # load the full image
                image_full = load_image(patient_id = pID,
                                        study_id   = sID,
                                        scan_name  = sNa, 
                                        datadir    = self.input_dir,
                                        dataset    = self.dataset,
                                        res        = self.res)

                if self.res == 'v':
                    infix = 'Low'
                else:
                    infix = 'High'
                coordinates = (int(row['annotation{}Resolution{}'.format(infix, i)]) for i in [1, 2, 3])
                # load the desired patch
                image = get_patch(image_full, coordinates, size=(40, 40, 40))

                # append image and labels
                images.append(image)

                if self.dataset == 'train':
                    label = row['histology']
                    # one hot enconding labels
                    if label == '2':
                        labels.append((1, 0, 0))
                    elif label == '20':
                        labels.append((0, 1, 0))
                    elif label == '21':
                        labels.append((0, 0, 1))

            except Exception as e:
                print('Failed reading idx {idx}...'.format(idx=i))
                print(e)

        # Assemble batch
        batch_x = np.stack(images).astype(K.floatx())
        if self.dataset == 'train':
            batch_y = np.stack(labels).astype(K.floatx())
        else:
            batch_y = None

        return batch_x, batch_y
