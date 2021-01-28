from keras.models import Model
from keras.layers import Activation, GlobalAveragePooling3D, AveragePooling3D, Input, Conv3D, Concatenate, MaxPooling3D, Dropout, Dense, Flatten

from keras import backend as K


def loss(y_true, y_pred):
    """
    Binary crossentropy as in: http://cs231n.stanford.edu/reports/2017/pdfs/23.pdf
    """
    N = y_true.shape
    print(N)
    L = 0
    for i in range(N[0]):
        L += y_true[i] * K.log(y_pred[i]) + (1 - y_true[i]) * K.log(1 - y_pred[i])
    return L / N


def create_SqueezeNet3D(patch_size=(40, 40, 40), drop_rate=0.2,
        loss_function='categorical_crossentropy', optimizer='sgd'):
    """ 
    Fully Convolutional Netowrk, from: http://cs231n.stanford.edu/reports/2017/pdfs/23.pdf
    """
    if not isinstance(patch_size, tuple):
        patch_size = (patch_size, patch_size, patch_size)
    K.clear_session() # destroy old graphs
    
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3
    
    in_x = Input(shape = (*patch_size, 1), name = 'input')
    x = Conv3D(96, 5, name = 'conv1')(in_x)
    x = MaxPooling3D(3, 2, name = 'maxpool1')(x)
    
    # fire module 2
    x = Conv3D(16, 1, padding='valid', name = 'fire2_squeeze')(x)
    x1 = Conv3D(64, 1, padding='valid', name = 'fire2_expand1')(x)
    x2 = Conv3D(64, 3, padding='same', name = 'fire2_expand2')(x)
    x = Concatenate(axis=channel_axis, name = 'concatenate_25')([x1, x2])
    
    # fire module 3
    x = Conv3D(16, 1, padding='valid', name = 'fire3_squeeze')(x)
    x1 = Conv3D(64, 1, padding='valid', name = 'fire3_expand1')(x)
    x2 = Conv3D(64, 3, padding='same', name = 'fire3_expand2')(x)
    x = Concatenate(axis=channel_axis, name = 'concatenate_26')([x1, x2])
    
    # fire module 4
    x = Conv3D(32, 1, padding='valid', name = 'fire4_squeeze')(x)
    x1 = Conv3D(128, 1, padding='valid', name = 'fire4_expand1')(x)
    x2 = Conv3D(128, 3, padding='same', name = 'fire4_expand2')(x)
    x = Concatenate(axis=channel_axis, name = 'concatenate_27')([x1, x2])
    x = MaxPooling3D(3, 2, name = 'maxpool4')(x)

    # fire module 5
    x = Conv3D(32, 1, padding='valid', name = 'fire5_squeeze')(x)
    x1 = Conv3D(128, 1, padding='valid', name = 'fire5_expand1')(x)
    x2 = Conv3D(128, 3, padding='same', name = 'fire5_expand2')(x)
    x = Concatenate(axis=channel_axis, name = 'concatenate_28')([x1, x2])
    
    # fire module 6
    x = Conv3D(48, 1, padding='valid', name = 'fire6_squeeze')(x)
    x1 = Conv3D(192, 1, padding='valid', name = 'fire6_expand1')(x)
    x2 = Conv3D(192, 3, padding='same', name = 'fire6_expand2')(x)
    x = Concatenate(axis=channel_axis, name = 'concatenate_29')([x1, x2])
    
    # fire module 7
    x = Conv3D(48, 1, padding='valid', name = 'fire7_squeeze')(x)
    x1 = Conv3D(192, 1, padding='valid', name = 'fire7_expand1')(x)
    x2 = Conv3D(192, 3, padding='same', name = 'fire7_expand2')(x)
    x = Concatenate(axis=channel_axis, name = 'concatenate_30')([x1, x2])
    
    # fire module 8
    x = Conv3D(64, 1, padding='valid', name = 'fire8_squeeze')(x)
    x1 = Conv3D(256, 1, padding='valid', name = 'fire8_expand1')(x)
    x2 = Conv3D(256, 3, padding='same', name = 'fire8_expand2')(x)
    x = Concatenate(axis=channel_axis, name = 'concatenate_31')([x1, x2])
    x = MaxPooling3D(3, 2, name = 'maxpool8')(x)
    
    # fire module 9
    x = Conv3D(64, 1, padding='valid', name = 'fire9_squeeze')(x)
    x1 = Conv3D(256, 1, padding='valid', name = 'fire9_expand1')(x)
    x2 = Conv3D(256, 3, padding='same', name = 'fire9_expand2')(x)
    x = Concatenate(axis=1, name = 'concatenate_32')([x1, x2])
    x = Dropout(drop_rate, name = 'fire9_dropout')(x)
    
    x = Conv3D(3, 1, name = 'conv10')(x)
    x = GlobalAveragePooling3D(name = 'avgpool10')(x)
    out_x = Activation('softmax', name = 'softmax')(x)
    
    model = Model(in_x, out_x)
    model.compile(optimizer, loss_function, metrics=['accuracy'])
    model.summary()
    return model
