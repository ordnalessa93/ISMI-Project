#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.layers import (Activation,
                          Input,
                          Conv3D,
                          MaxPooling3D,
                          Dropout,
                          Dense,
                          Flatten,
                          Concatenate,
                          GlobalAveragePooling3D,
                         )
from keras.models import Model


def create_initial_model(patch_size=(40, 40, 40), drop_rate=0.2):
    """
    Creates a Neural Network that does 3d convolution on a object of shape The
    rate of dropout can be regularized by settign drop_rate. This is by default
    set to 0.2 patch_size should be an integer or a tuple of 3 integers and is
    by default a cube of 40 by 40 by 40. should not be lower than 38.
    """
    if not isinstance(patch_size, tuple):
        patch_size = (patch_size, patch_size, patch_size)

    # Creating the model
    m_in = Input(shape=(*patch_size, 1))
    # Convolution part
    layer = Conv3D(64, 3, activation='relu', kernel_initializer='he_normal')(m_in)
    layer = Dropout(drop_rate)(layer)
    for i in range(3):
        for j in range(2):
            layer = Conv3D(64*(2**i), 3, activation='relu', kernel_initializer='he_normal')(layer)
            layer = Dropout(drop_rate)(layer)
        layer = MaxPooling3D()(layer)
    # fully connected
    layer = Flatten()(layer)
    layer = Dense(128, activation='relu')(layer)
    layer = Dropout(drop_rate)(layer)
    m_out = Dense(3, activation='softmax')(layer)
    model = Model(m_in, m_out)
    return model


def create_second_model(patch_size=(40, 40, 40), drop_rate=0.2):
    if not isinstance(patch_size, tuple):
        patch_size = (patch_size, patch_size, patch_size)

    m_in = Input(shape=(*patch_size, 1))

    x = Conv3D(64, 3, activation='relu', kernel_initializer='he_normal')(m_in)
    x = Conv3D(64, 3, activation='relu', kernel_initializer='he_normal')(x)
    x = MaxPooling3D()(x)
    x = Dropout(drop_rate)(x)

    x = Conv3D(128, 3, activation='relu', kernel_initializer='he_normal')(m_in)
    x = Conv3D(128, 3, activation='relu', kernel_initializer='he_normal')(x)
    x = MaxPooling3D()(x)
    x = Dropout(drop_rate)(x)

    x = Conv3D(256, 3, activation='relu', kernel_initializer='he_normal')(m_in)
    x = Conv3D(256, 3, activation='relu', kernel_initializer='he_normal')(x)
    x = MaxPooling3D()(x)
    x = Dropout(drop_rate)(x)

    x = GlobalAveragePooling3D()(x)
    x = Dense(1024, activation='relu', kernel_initializer='he_normal')(x)
    x = Dropout(drop_rate)(x)
    x = Dense(512, activation='relu', kernel_initializer='he_normal')(x)
    x = Dropout(drop_rate)(x)
    m_out = Dense(3, activation='softmax', kernel_initializer='he_normal')(x)

    model = Model(m_in, m_out)
    return model


def create_squeezenet3d_model(patch_size=(40, 40, 40), drop_rate=0.2):
    """
    Fully Convolutional Network, from:
    http://cs231n.stanford.edu/reports/2017/pdfs/23.pdf
    """
    if not isinstance(patch_size, tuple):
        patch_size = (patch_size, patch_size, patch_size)

    in_x = Input(shape=(*patch_size, 1), name='input')
    x = Conv3D(64, 3, activation='relu', name='conv1')(in_x)
    x = MaxPooling3D(3, 2, name='maxpool1')(x)

    # fire module 2
    x = Conv3D(16, 1, padding='valid', activation='relu', name='fire2_squeeze')(x)
    x1 = Conv3D(64, 1, padding='valid', activation='relu', name='fire2_expand1')(x)
    x2 = Conv3D(64, 3, padding='same', activation='relu', name='fire2_expand2')(x)
    x = Concatenate(name='concatenate_25')([x1, x2])

    # fire module 3
    x = Conv3D(16, 1, padding='valid', activation='relu', name='fire3_squeeze')(x)
    x1 = Conv3D(64, 1, padding='valid', activation='relu', name='fire3_expand1')(x)
    x2 = Conv3D(64, 3, padding='same', activation='relu', name='fire3_expand2')(x)
    x = Concatenate(name='concatenate_26')([x1, x2])
    x = MaxPooling3D(3, 2, name='maxpool3')(x)

    # fire module 4
    x = Conv3D(32, 1, padding='valid', activation='relu', name='fire4_squeeze')(x)
    x1 = Conv3D(128, 1, padding='valid', activation='relu', name='fire4_expand1')(x)
    x2 = Conv3D(128, 3, padding='same', activation='relu', name='fire4_expand2')(x)
    x = Concatenate(name='concatenate_27')([x1, x2])

    # fire module 5
    x = Conv3D(32, 1, padding='valid', activation='relu', name='fire5_squeeze')(x)
    x1 = Conv3D(128, 1, padding='valid', activation='relu', name='fire5_expand1')(x)
    x2 = Conv3D(128, 3, padding='same', activation='relu', name='fire5_expand2')(x)
    x = Concatenate(name='concatenate_28')([x1, x2])
    x = MaxPooling3D(3, 2, name='pool5')(x)

    # fire module 6
    x = Conv3D(48, 1, padding='valid', activation='relu', name='fire6_squeeze')(x)
    x1 = Conv3D(192, 1, padding='valid', activation='relu', name='fire6_expand1')(x)
    x2 = Conv3D(192, 3, padding='same', activation='relu', name='fire6_expand2')(x)
    x = Concatenate(name='concatenate_29')([x1, x2])

    # fire module 7
    x = Conv3D(48, 1, padding='valid', activation='relu', name='fire7_squeeze')(x)
    x1 = Conv3D(192, 1, padding='valid', activation='relu', name='fire7_expand1')(x)
    x2 = Conv3D(192, 3, padding='same', activation='relu', name='fire7_expand2')(x)
    x = Concatenate(name='concatenate_30')([x1, x2])

    # fire module 8
    x = Conv3D(64, 1, padding='valid', activation='relu', name='fire8_squeeze')(x)
    x1 = Conv3D(256, 1, padding='valid', activation='relu', name='fire8_expand1')(x)
    x2 = Conv3D(256, 3, padding='same', activation='relu', name='fire8_expand2')(x)
    x = Concatenate(name='concatenate_31')([x1, x2])
    x = MaxPooling3D(3, 2, name='maxpool8')(x)

    # fire module 9
    x = Conv3D(64, 1, padding='valid', activation='relu', name='fire9_squeeze')(x)
    x1 = Conv3D(256, 1, padding='valid', activation='relu', name='fire9_expand1')(x)
    x2 = Conv3D(256, 3, padding='same', activation='relu', name='fire9_expand2')(x)
    x = Concatenate(axis=1, name='concatenate_32')([x1, x2])

    x = Dropout(drop_rate, name='fire9_dropout')(x)

    x = Conv3D(3, 1, activation='relu', name='conv10')(x)
    x = GlobalAveragePooling3D(name='avgpool10')(x)
    out_x = Activation(activation='softmax')(x)

    model = Model(in_x, out_x)
    return model
