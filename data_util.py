#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.models import load_model
from os.path import join
import SimpleITK as sitk
import keras.backend as K
import numpy as np
import pandas as pd

def get_patch(image, coord, size=(70, 70, 40), shifted=True):
    '''
    Function that returns a fixed sized patch, centered around a coordinate,
    from a given numpy array.
    Note: The input coordinates should be a tuple in (y, x, z) format.
    '''
    if type(size) == int:
        size = (size, size, size)
    
    patch  = np.zeros((size[1], size[0], size[2])) # numpy array of the patch
    offset = (size[1]//2, size[0]//2, size[2]//2) # offset
    
    #image = np.pad(image, 30, 'constant')
    
    coordinates = np.array(list(coord))        # list of coordinates
    img_shape   = np.array(image.shape)        # shape of the image
    
    # change representation (x,y,z) to (y,x,z) 
    img_shape[[0,1,2]] = img_shape[[1,0,2]]
    coordinates[[0,1,2]] = coordinates[[1,0,2]]
    
    # for each dimension shift the coordinates (only if needed)
    if shifted is True:
        for i in range(3):
            print(coordinates[i], offset[i])
            if coordinates[i] + offset[i] >= img_shape[i]:
                if size[i] % 2 != 0:
                    coordinates[i] = img_shape[i] - offset[i] - 1
                else:
                    coordinates[i] = img_shape[i] - offset[i]

            if coordinates[i] - offset[i] < 0:
                coordinates[i] = 0 + offset[i]

    # get the coordinates
    x, y, z = coordinates
        
    # Define start point and padding size before start
    xstart = x-offset[0]
    xpad = max(0, xstart-(x-offset[0]))
    ystart = y-offset[1]
    ypad = max(0, ystart-(y-offset[1]))
    zstart = z-offset[2]
    zpad = max(0, zstart-(z-offset[2]))

    # Copy image over patch
    for i in range(size[1]):
        x = xstart+i
        for j in range(size[0]):
            y = ystart+j
            for k in range(size[2]):
                z = zstart+k
                if x>image.shape[0]-1 or y>image.shape[1]-1 or z>image.shape[2]-1: # to big, so needs padding
                    patch[i,j,k]=0
                elif x < xpad or y < ypad or z < zpad:
                    patch[i,j,k]=0
                else:
                    patch[i,j,k] = image[x,y,z]
    
    return np.expand_dims(patch, -1)


def load_image(patient_id, study_id, scan_name, datadir='./data',
               dataset='train', res='v', verbose=0):
    '''
    Load MHD file given some information. patient_id, study_id and scan_name
    should be strings from the corresponding values in the CSV file. It is
    important that patient_id has the prefixed 0 in it; to make sure, load the
    CSV file with dtype=str (i.e. pd.read_csv(csvpath, dtype=str)).

    res should be either 'v' for low-res or 't' for hi-res image. If verbose >
    0, this function will print the path of the image being loaded for
    debugging purposes.
    '''
    if dataset not in ['train', 'test']:
        raise ValueError('dataset can only be train or test')
    if res not in ['v', 't']:
        raise ValueError('res can only be v (low-res) or t (hi-res)')

    # Create path string
    fname = '{res}{patient_id}{study_num}{scan_name}.mhd'.format(
                res=res,
                patient_id=patient_id,
                study_num=study_id[2:],
                scan_name=scan_name
            )
    img_path = join(datadir, dataset, patient_id, study_id, fname)

    if verbose > 0:
        print('Loading file: {}'.format(img_path))

    img = sitk.ReadImage(img_path)
    img_array = sitk.GetArrayFromImage(img)
    
    # Normalize to 0-1
    if res == 'v':
        img_array = img_array / 4018.
    else:
        img_array = img_array / 255.
    # Change axes from (z, y, x) to (y, x, z)
    return np.moveaxis(img_array, 0, -1)


def create_submission(test_df, test_seq, model_files, data_dir = './data',
                      result_dir = './'):
    '''
    Function that generates a submission taking as an imput the corresponding test
    dataframe, the name of the best model to be loaded, the directory of the data
    and the directory of the model
    '''

    print('Evaluating test set ...')

    predictions = []
    for name in model_files:
        K.clear_session()
        # load the best model
        best_model = load_model(name)

        # run the model over the test samples
        y_pred = best_model.predict_generator(test_seq,
                                              use_multiprocessing=True,
                                              workers=4,
                                              verbose=1)
        predictions.append(y_pred)

    predictions = np.stack(predictions, axis=-1)
    predictions = predictions.mean(axis=-1)

    # get the argument with the higher confidence score
    y_pred = np.argmax(predictions, axis=1)

    # convert the indices in the correct labels
    convert = {0: 2, 1: 20, 2: 21}

    # store the cases and corresponding class
    column_names = ['case', 'class']
    values = []
    for i in range(len(test_df)):
        case = test_df.iloc[i]['grandChallangeCaseName']
        values.append((case, convert[y_pred[i]]))

    # create DataFrame with the values and export it to ".csv" format
    df = pd.DataFrame(data=values, columns=column_names)
    df.to_csv(join(result_dir, 'submission.csv'), index=None)


def create_submission_2step(test_df, test_seq, model_files_1st, model_files_2nd,
                            data_dir = './data',result_dir = './'):
    '''
    Function that generates a submission taking as an imput the corresponding test
    dataframe, the name of the best model to be loaded, the directory of the data
    and the directory of the model
    '''

    print('Evaluating test set ...')

    predictions_1st = []
    for name in model_files_1st:
        K.clear_session()
        # load the best model
        best_model = load_model(name)

        # run the model over the test samples
        y_pred = best_model.predict_generator(test_seq,
                                              use_multiprocessing=True,
                                              workers=4,
                                              verbose=1)
        predictions_1st.append(y_pred)

    predictions_1st = np.stack(predictions_1st, axis=-1)
    predictions_1st = predictions_1st.mean(axis=-1)

    # get the argument with the higher confidence score
    y_pred_1st = np.argmax(predictions_1st, axis=1)

    # ----------------------------------------------------
    
    predictions_2nd = []
    for name in model_files_2nd:
        K.clear_session()
        # load the best model
        best_model = load_model(name)

        # run the model over the test samples
        y_pred = best_model.predict_generator(test_seq,
                                              use_multiprocessing=True,
                                              workers=4,
                                              verbose=1)
        predictions_2nd.append(y_pred)

    predictions_2nd = np.stack(predictions_2nd, axis=-1)
    predictions_2nd = predictions_2nd.mean(axis=-1)

    # get the argument with the higher confidence score
    y_pred_2nd = np.argmax(predictions_2nd, axis=1)    
    
    # --------------------------------------------------------
    
    # convert the indices in the correct labels
    convert = {0: 21, 1:{0: 20, 1:2}}

    # store the cases and corresponding class
    column_names = ['case', 'class']
    values = []
    for i in range(len(test_df)):
        case = test_df.iloc[i]['grandChallangeCaseName']
        if y_pred_1st[i] == 0:
            values.append((case, convert[y_pred_1st[i]]))
        else:
            values.append((case, convert[y_pred_1st[i]][y_pred_2nd[i]]))

    # create DataFrame with the values and export it to ".csv" format
    df = pd.DataFrame(data=values, columns=column_names)
    df.to_csv(join(result_dir, 'submission.csv'), index=None)
