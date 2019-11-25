import numpy as np
import pandas as pd
from scipy.ndimage.interpolation import shift
import tensorflow as tf

def read(dir_, cls_, id_, lc_params):
    '''Function to read the files using all cores
    dir_: address of one light curve
    cls_: class of the object
    col_names: column names for each light curve. They must indicate
            [time, magnitude, errror]'''


    col_names= lc_params['usecols']
    # Read the dataframe for an object
    df_lc = pd.read_csv(dir_, **lc_params)

    # Some records are bad, remove all of them, and repeated measurements
    duplicates = -df_lc.duplicated(subset=col_names[0], keep='first')
    df_lc = df_lc[duplicates]

    # Sort the values
    df_lc = df_lc.sort_values([col_names[0]], inplace=False)

    # T ransform into a np.array
    # Exclude the error column 2
    lc = df_lc.values[:, [0, 1]]

    # Only leave good records
    # good_rec = lc[:, 0] > 0
    # lc = lc[good_rec, :]

    return lc, cls_, id_

def process(cls_, _data_, w, s, w_time):
    '''Process one single element, to be excecuted in paralell.'''

    ID = _data_[1]
    data_ = _data_[0]

    # Make the differences
    d = data_ - shift(data_, [1, 0], cval=0)
    # Remove the first measument because it is unchanged
    d = d[1:]

    # Place the data in each window
    N = int((d.shape[0]-w+s)/s)
    X = [d[i*s:i*s+w, 1] for i in range(N)]
    T = [d[i*s:i*s+w, 0] for i in range(N)]
    try:
        if w_time:
            matrices = np.concatenate((T, X), axis=1)
        else:
            matrices = np.asarray(X)
    except Exception as e:
        print(e)
        print(cls_,data_)

    return cls_, matrices, ID

def serialize(sequence, label, ID_):


    N = sequence.shape[0]
    ID_ = str(ID_).encode()

    # Non-sequential features of our example
    element_context = tf.train.Features(feature={
    'N': tf.train.Feature(int64_list=tf.train.Int64List(value=[N])),
    'ID': tf.train.Feature(bytes_list=tf.train.BytesList(value=[ID_])),
    'Label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    })

    # Feature lists the sequential feature of our example
    lc_1 = tf.train.Feature(
      float_list=tf.train.FloatList(
        value=sequence.ravel()))

    lcs = tf.train.FeatureList(feature=[lc_1])

    lc_dict = {
      'LightCurve_1': lcs
    }

    element_lists = tf.train.FeatureLists(feature_list=lc_dict)
    # The object we return
    example = tf.train.SequenceExample(
        context=element_context,
        feature_lists=element_lists)

    return example
