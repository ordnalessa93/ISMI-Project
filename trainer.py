#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
from generators import PatchGenerator, PatchSequence
from keras.callbacks import Callback
from keras.models import load_model
from keras.optimizers import Adam, RMSprop
from keras.utils import multi_gpu_model
from networks import (create_initial_model,
                      create_second_model,
                      create_squeezenet3d_model,
                     )
from skimage.transform import rotate
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
import keras.backend as K
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import warnings

NETWORKS = {
    'initial': create_initial_model,
    'second': create_second_model,
    'squeezenet3d': create_squeezenet3d_model,
}

LOSS_FUNCTION = 'categorical_crossentropy'

OPTIMIZERS = {
    'adam': Adam,
    'rmsprop': RMSprop,
}


class MultiGPUCheckpoint(Callback):

    def __init__(self, filename, verbose=0):
        super().__init__()
        self.filename = filename
        self.verbose = verbose
        self.val_accs = []

    def on_epoch_end(self, epoch, logs=None):
        if not self.val_accs:
            self.model.layers[-2].save(self.filename)
        elif logs['val_acc'] > max(self.val_accs):
            if self.verbose > 0:
                print('Saving to {}'.format(self.filename))
            self.model.layers[-2].save(self.filename)
        self.val_accs.append(logs['val_acc'])


class Accuracies(Callback):

    def __init__(self, valid_seq):
        super().__init__()
        self.valid_seq = valid_seq
        self.label_accuracies = []

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict_generator(self.valid_seq,
                                              workers=4,
                                              use_multiprocessing=True)
        y_pred = np.argmax(y_pred, axis=1)

        y_true = self.valid_seq.get_all_labels()
        y_true[y_true == 2] = 0
        y_true[y_true == 20] = 1
        y_true[y_true == 21] = 2

        cm = confusion_matrix(y_true, y_pred)
        ps = cm.diagonal() / cm.sum(axis=1)
        self.label_accuracies.append(ps)


def create_model(network, optimizer, drop_rate, multi_gpu):
    orig_model = NETWORKS[network](drop_rate=drop_rate)
    if multi_gpu:
        parallel_model = multi_gpu_model(orig_model, cpu_relocation=True)
        parallel_model.compile(optimizer=optimizer, loss=LOSS_FUNCTION,
                               metrics=['accuracy'])
    else:
        orig_model.compile(optimizer=optimizer, loss=LOSS_FUNCTION,
                           metrics=['accuracy'])
        parallel_model = None
    return orig_model, parallel_model


def create_optimizer(name, lr, decay):
    return OPTIMIZERS[name](lr=lr, decay=decay)


def make_augmentation_func(aug, aug_hflip, aug_vflip, aug_rotate,
                           aug_brightness):
    if not aug:
        return None

    def augf(img):
        if np.random.random() > aug and aug_hflip:
            img = np.fliplr(img)
        if np.random.random() > aug and aug_vflip:
            img = np.flipud(img)
        if np.random.random() > aug and aug_rotate:
            tmp = np.squeeze(img)
            angle = np.random.uniform(0, aug_rotate)
            tmp = rotate(tmp, angle)
            img = np.expand_dims(tmp, -1)
        if np.random.random() > aug and aug_brightness:
            up_delta = 1. - img.max()
            down_delta = img.min()
            delta = min(up_delta, down_delta)
            img = img + np.random.uniform(-delta, delta)
        return img[15:55, 15:55, ...]

    return augf


def make_generators(csv, train_patients, validation_patients, batch_size,
                    augf):
    train_csv = csv.loc[csv['patientID'].isin(train_patients), :]
    valid_csv = csv.loc[csv['patientID'].isin(validation_patients), :]

    train_gen = PatchGenerator(
        input_dir='./data',
        dataframe=train_csv,
        batch_size=batch_size,
        augmentation_fn=augf
    )

    valid_seq = PatchSequence(
        input_dir='./data',
        dataframe=valid_csv,
        batch_size=batch_size
    )

    return train_gen, valid_seq


def train_model(args):
    csv = pd.read_csv('./data/trainingSet.csv', dtype=str)

    # Create patient K-folder
    unique_patients = csv.patientID.unique()
    kf = KFold(5, shuffle=True, random_state=42)
    folds = kf.split(unique_patients)

    # Make augmentation function
    augf = make_augmentation_func(args.aug,
                                  args.aug_hflip,
                                  args.aug_vflip,
                                  args.aug_rotate,
                                  args.aug_brightness)

    if args.class_weight:
        cw = compute_class_weight('balanced',
                                  np.unique(csv['histology'].values),
                                  csv['histology'].values)
        cwdict = dict(enumerate(cw))
    else:
        cwdict = None

    accuracies = []
    for i, (train_idxs, val_idxs) in enumerate(folds, start=1):
        K.clear_session()
        print('Fold {}'.format(i))

        train_patients = unique_patients[train_idxs]
        val_patients = unique_patients[val_idxs]

        train_gen, valid_seq = make_generators(csv,
                                               train_patients,
                                               val_patients,
                                               args.batch_size,
                                               augf)
        optimizer = create_optimizer(args.optimizer, args.lr, args.decay)
        orig_net, parallel_net = create_model(args.network, optimizer,
                                              args.drop_rate,
                                              args.multi_gpu)

        save_filename = '{}_fold_{}.h5'.format(args.filename, i)
        if args.multi_gpu:
            cp = MultiGPUCheckpoint(save_filename, verbose=1)
        else:
            cp = ModelCheckpoint(save_filename, save_best_only=True, verbose=1,
                                 monitor='val_acc')
        ps = Accuracies(valid_seq)

        train_model = parallel_net or orig_net
        results = train_model.fit_generator(train_gen,
                                            steps_per_epoch=len(train_gen),
                                            validation_data=valid_seq,
                                            epochs=args.epochs,
                                            use_multiprocessing=True,
                                            workers=4,
                                            class_weight=cwdict,
                                            callbacks=[cp, ps],
                                            verbose=1)

        h = results.history
        plt.figure()
        plt.plot(h['loss'])
        plt.plot(h['acc'])
        plt.plot(h['val_loss'])
        plt.plot(h['val_acc'])
        plt.legend(['loss', 'acc', 'val_loss', 'val_acc'])
        plt.savefig('{}.traininglog.png'.format(save_filename))

        y_true = valid_seq.get_all_labels()
        y_true[y_true == 2] = 0
        y_true[y_true == 20] = 1
        y_true[y_true == 21] = 2
        best_net = load_model(save_filename)
        y_pred = best_net.predict_generator(valid_seq,
                                            workers=4,
                                            use_multiprocessing=True)
        y_pred = np.argmax(y_pred, axis=1)
        cm = confusion_matrix(y_true, y_pred)

        plt.figure()
        sns.heatmap(cm, annot=True)
        plt.savefig('{}.confusionmatrix.png'.format(save_filename))

        precs = np.array(ps.label_accuracies)

        plt.figure()
        for i in range(precs.shape[1]):
            plt.plot(precs[:, i])
        plt.legend(['0', '1', '2'])
        plt.savefig('{}.accuracies.png'.format(save_filename))

        accuracies.append(max(h['val_acc']))

    with open('{}_score.txt'.format(args.filename), 'w') as f:
        print('Mean accuracy: {:.4f}'.format(np.mean(accuracies)), file=f)


if __name__ == '__main__':
    p = ArgumentParser('Experiment utility')

    # Network variables
    p.add_argument('--network', required=True, choices=NETWORKS.keys())
    p.add_argument('--drop-rate', type=float, default=.2)
    p.add_argument('--multi-gpu', action='store_true')

    # Training variables
    p.add_argument('--epochs', required=True, type=int)
    p.add_argument('--batch-size', required=True, type=int)
    p.add_argument('--class-weight', action='store_true')
    p.add_argument('--filename', required=True, type=str)

    # Optimizer variables
    p.add_argument('--optimizer', required=True, choices=OPTIMIZERS.keys())
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--decay', type=float, default=.0)

    # Augmentation variables
    p.add_argument('--aug', type=float)
    p.add_argument('--aug-hflip', action='store_true')
    p.add_argument('--aug-vflip', action='store_true')
    p.add_argument('--aug-rotate', type=float)
    p.add_argument('--aug-brightness', action='store_true')

    train_model(p.parse_args())
    sys.exit(0)
