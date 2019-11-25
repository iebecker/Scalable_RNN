from joblib import Parallel, delayed
import json
import numpy as np
from numpy.random import shuffle
import os
import pandas as pd
from scipy.ndimage.interpolation import shift
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tqdm import tqdm
import utils

class Preprocesser():
    '''Class that implements functions prepare, read,
    transform and save the data'''
    def __init__(self, max_L=40000, min_L=500, min_N=0, max_N = 2000, w=4, s=2
            , w_time=True, num_cores=8, lc_parameters = None):

        # We already imposed a minimum number of poitns per light curve
        # Impose a minimum of points per light curve
        self.max_L = max_L
        self.min_L = min_L
        self.max_N = max_N
        self.min_N = min_N

        # Container for the data
        self.Labels = []
        self.Matrices = []
        self.IDs = []

        # Parameters for the objects
        self.w = w
        self.s = s
        self.w_time = w_time
        self.njobs = num_cores


        # Parameters to read each light curve
        self.lc_parameters = lc_parameters

        # Auxiliary functions to parallelize
        # Parallel does not work with class functions (for some reason)
        self.__func_read = utils.read
        self.__func_process = utils.process
        self.__func_serialize = utils.serialize

    def set_execution_variables(self, file_train, save_dir
                    , train_size, val_size, test_size
                    , inference, inference_folder):

        '''Defines paths and split information.
        This function separates the object itself with the different
        excecutions of the object.'''

        # Set inference mode
        self.inference = inference
        self.inference_folder = inference_folder
        self.save_dir = save_dir

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.default_split = True
        # Dataset info
        self.file_train = file_train

        # Splits fractions
        self.train_size = train_size
        self.test_size = test_size
        self.val_size = val_size

        # Set the train/test/val sizes
        # If train_size are files, with the same format as
        #"file_train", use these to create the splits.
        if( type(train_size)==str and type(test_size)==str and type(val_size)==str):
            self.default_split = False
            self.file_train = train_size
            self.file_test = test_size
            self.file_val = val_size

        # Read the datasets to be used. Just one if default splits, three if custom splits.
        self.read_datasets()

        # Extract classes and the number of them
        self.classes = list(set(self.data_train.Class))
        self.num_classes = len(self.classes)

        # Dictionary to transform string labels to ints
        self.trans = {c: n for c, n in zip(self.classes, range(self.num_classes))}
        self.trans_inv = dict(zip(self.trans.values(), self.trans.keys()))

        # If in inference mode, override min,max_N, and min,max_L
        if self.inference:
            self.max_L = 1e9 #Arbitrary
            self.min_L = 0
            self.min_N = 0
            self.max_N = 1e9 #Arbitrary

    def filter_train(self):
        '''Filter the objects to be read.
        First by imposing restriction to the number of data points.
        Second, by extracting a random sample of up uo max_L elements
        per category.'''
        # Objects that fulfill the number of datapoints condition
        bol21 = self.data_train.N >=self.min_N
        bol22 = self.data_train.N <=self.max_N
        bol2 = np.logical_and(bol21,bol22)
        # Leave up_to N_max objects per class
        dfs = []
        for i in self.classes.copy():
            # Objects of the class
            bol1 = self.data_train.Class == i
            # Both conditions
            bol = np.logical_and(bol1,bol2)
            sel = self.data_train[bol]

            # Limit the minimum number of light curves
            if sel.shape[0] < self.min_L:
                # Update the classes
                self.classes.remove(i)
                self.num_classes = len(self.classes)
                # Skip the class
                continue

            # Random sample of objects, not done in inference
            if not self.inference:
                # Return the min among the number of objects and max_L
                num = min(self.max_L, sel.shape[0])
                # Get a random sample
                sel = sel.sample(num, replace=False, axis=0)
            dfs.append(sel)
        # Join the dataframes of each class together
        self.data_train = pd.concat(dfs)


    def read_datasets(self):
        '''Read the dataset, extract the LCs information, with the class and ID.
        Filter the specified number of LC per class, so it does not read everything.

        If default_split is False, each one of the folds is read separately.
        No further filtering is done.'''

        if self.default_split:
            # Read stars Data
            self.data_train = pd.read_csv(self.file_train, usecols=['ID', 'Address','Class','N'])

            # Extract classes and the number of them
            self.classes = list(set(self.data_train.Class))
            self.num_classes = len(self.classes)

            # Filter train according to number of observations and elements per class
            self.filter_train()
        else:
            self.data_train= pd.read_csv(self.file_train, usecols=['ID', 'Address','Class','N'])
            self.data_test = pd.read_csv(self.file_test, usecols=['ID', 'Address','Class','N'])
            self.data_val = pd.read_csv(self.file_val, usecols=['ID', 'Address','Class','N'])

    def __parallel_read_util(self, _data_):
        '''Reads un parallel light curves in _data_.'''
        ext = Parallel(self.njobs)(delayed(self.__func_read)(address_, class_, id_, self.lc_parameters) for address_, class_, id_ in
                                 tqdm(zip(_data_.Address, _data_.Class, _data_.ID)))
        return ext

    def __sort_lcs_util(self, read_lcs):
        '''Create a dictionary, where each class is the key, the id and
        light curve itself are stored in a list, as values.'''

        # Create a dictionary by class
        lcs = {c: [] for c in self.classes}
        # For each class, light curve i[0] and the id i[2]
        [lcs[i[1]].append([i[0],i[2]]) for i in read_lcs]
        return lcs

    def parallel_read_default(self):
        '''Read the data using default splits and preprocessing.
        Store all the lcs in a dict where the classes are keys.'''

        # Read the light curves in parallel
        print('Reading')
        read_lcs = self.__parallel_read_util(self.data_train)

        # ext = Parallel(self.njobs)(delayed(self.__func_read)(address_, class_, id_, self.lc_parameters) for address_, class_, id_ in
        #                          tqdm(zip(self.data_train.Address, self.data_train.Class, self.data_train.ID)))
        # Store the light curves by class and ID info.
        self.lcs = self.__sort_lcs_util(read_lcs)
        # Create a dictionary by class
        # self.lcs = {c: [] for c in self.classes}
        # # For each class, light curve i[0] and the id i[2]
        # [self.lcs[i[1]].append([i[0],i[2]]) for i in read_lcs]

    def parallel_read_custom(self):
        '''Read the data using custom splits and preprocessing.
        This do no applies any filter to the data.
        Store all the lcs in a dictS where the classes are keys.
        One dict per split. Only train test and val are used.'''

        # Read the light curves in parallel
        print('Reading Train')
        read_lcs_train = self.__parallel_read_util(self.data_train)
        print('Reading Test')
        read_lcs_test = self.__parallel_read_util(self.data_test)
        print('Reading Val')
        read_lcs_val = self.__parallel_read_util(self.data_val)

        # Store them
        self.lcs_train = self.__sort_lcs_util(read_lcs_train)
        self.lcs_test = self.__sort_lcs_util(read_lcs_test)
        self.lcs_val = self.__sort_lcs_util(read_lcs_val)


    def parallel_read(self):
        '''Run parallel read using n_jobs threads, depending on the user choice.'''
        if self.default_split:
            self.parallel_read_default()
        else:
            self.parallel_read_custom()

    def parallel_process(self):
        '''Extracts the data and transform it into matrix representation.'''
        if self.default_split:
            self.parallel_process_default()
        else:
            self.parallel_process_custom()

    def parallel_process_custom(self):
        '''Extracts the data into matrix representation, maintaininig the folds
        defined by the user.'''
        print('Processing')
        # Process the data
        self.Labels_train, self.Matrices_train, self.IDs_train = self.__process_lcs_util(self.lcs_train)
        self.Labels_test, self.Matrices_test, self.IDs_test = self.__process_lcs_util(self.lcs_test)
        self.Labels_val, self.Matrices_val, self.IDs_val = self.__process_lcs_util(self.lcs_val)

        # Shuffle the data
        if not self.inference:
            self.Labels_train, self.Matrices_train, self.IDs_train = self.__process_shuffle_util(self.Labels_train, self.Matrices_train, self.IDs_train)

    def __process_shuffle_util(self, _labels_, _matrices_, _ids_):
        '''Shuffles the data.'''

        ind = np.arange(len(_labels_))
        shuffle(ind)
        # Aplly the shuffle dindices
        _labels_ = _labels_[ind]
        # self.Ns = self.Ns[self.ind]
        _matrices_ = _matrices_[ind]
        _ids_ = _ids_[ind]
        return _labels_, _matrices_, _ids_

    def __process_lcs_util(self, lcs):
        '''Fucntion to process the lcs given lcs.'''
        Labels = []
        Matrices = []
        IDs = []
        for c in self.classes:
            sel = lcs[c]
            # Run the process function in parallel
            processed = Parallel(self.njobs)(delayed(self.__func_process)(c, l, self.w, self.s, self.w_time) for l in tqdm(sel))

            _Labels, _Matrices, _IDs = list(zip(*processed))
            # Store in list the information.
            # The order is preserved, so an jth element in all lists will correspond to the same object
            # Change the class to a number, and store it into a list
            _Labels = [self.trans[i] for i in _Labels]
            # Store the matrix representation into a list
            _Matrices = [i.astype(np.float32) for i in _Matrices]
            # Store the IDs into a list
            Labels.append(_Labels)
            Matrices.append(_Matrices)
            IDs.append(_IDs)

        Labels = np.concatenate(Labels, axis=0)
        Matrices = np.concatenate(Matrices, axis=0)
        IDs = np.concatenate(IDs, axis=0)

        return Labels, Matrices, IDs

    def parallel_process_default(self):
        print('Processing')

        # Process the data
        self.Labels, self.Matrices, self.IDs = self.__process_lcs_util(self.lcs)

        # Shuffle the data
        if not self.inference:
            self.Labels, self.Matrices, self.IDs = self.__process_shuffle_util(self.Labels, self.Matrices, self.IDs)

    def indices(self, train_ids, val_ids, test_ids):
        ind = range(len(self.IDs))
        ind_dict = dict(zip(self.IDs, ind))

        ind_train = list(map(ind_dict.get, train_ids))
        ind_val = list(map(ind_dict.get, val_ids))
        ind_test = list(map(ind_dict.get, test_ids))

        return ind_train, ind_test, ind_val

    def cls_metadata(self, labels):
        keys, values = np.unique(labels, return_counts= True)
        values_norm = values/sum(values)
        values_norm = ['({0:.3f} %)'.format(100*v) for v in values_norm]
        values = [str(v1)+' '+v2 for v1,v2 in zip(values, values_norm)]
        keys = [self.trans_inv[k] for k in keys]
        hist = dict(zip(keys, values))
        hist['Total'] = len(labels)
        return hist

    def split_train(self):
        self.data_train= self.data_train.set_index('ID')

        train_ids, test_val_ids = train_test_split(
            self.data_train.index.values,
            train_size=self.train_size,
            stratify=self.data_train.Class)

        test_ids, val_ids = train_test_split(
            test_val_ids,
            train_size=self.test_size/(1-self.train_size),
            stratify=self.data_train.loc[test_val_ids].Class)

        ind_train, ind_test, ind_val = self.indices(train_ids, val_ids, test_ids)

        self.Matrices_train = self.Matrices[ind_train]
        self.Labels_train = self.Labels[ind_train]
        self.IDs_train = self.IDs[ind_train]

        self.Matrices_val = self.Matrices[ind_val]
        self.Labels_val = self.Labels[ind_val]
        self.IDs_val = self.IDs[ind_val]

        self.Matrices_test = self.Matrices[ind_test]
        self.Labels_test = self.Labels[ind_test]
        self.IDs_test = self.IDs[ind_test]

        splits_labels = [self.Labels_train, self.Labels_test, self.Labels_val]

        # values = [self.cls_metadata(labels) for labels in splits_labels]
        # keys = ['Train set', 'Test set', 'Val set']
        # metadata = dict(zip(keys, values))
        # metadata['Keys'] = self.trans_inv
        # self.splits_metadata = metadata

    def get_metadata_split(self):
        '''Get the metadata of each splits.'''

        splits_labels = [self.Labels_train, self.Labels_test, self.Labels_val]

        values = [self.cls_metadata(labels) for labels in splits_labels]
        keys = ['Train set', 'Test set', 'Val set']
        metadata = dict(zip(keys, values))
        metadata['Keys'] = self.trans_inv
        self.splits_metadata = metadata


    def serialize_all(self):
        '''Serialize the data into TFRecords.'''

        self.serialize(self.Matrices_train,
                       self.Labels_train,
                       self.IDs_train,
                       self.save_dir+'Train.tfrecord')

        self.serialize(self.Matrices_val,
                       self.Labels_val,
                       self.IDs_val,
                       self.save_dir+'Val.tfrecord')

        self.serialize(self.Matrices_test,
                       self.Labels_test,
                       self.IDs_test,
                       self.save_dir+'Test.tfrecord')

    def serialize_inference(self, save_path=None):
        '''Serialize data for inference.'''
        if save_path is None:
            save_path = self.inference_folder+'Inference.tfrecord'

        self.serialize(self.Matrices[i],
                        self.Labels[i],
                        self.IDs[i],
                        save_path)

    def serialize(self, Matrices, Labels, IDs, save_path):
        '''Serialize objects given the data and path.'''
        with open(save_path, 'w') as f:
            writer = tf.io.TFRecordWriter(f.name)
            for i in range(len(IDs)):
                ex = self.__func_serialize(Matrices[i], Labels[i], IDs[i])
                writer.write(ex.SerializeToString())

    def write_metadata_process(self):
        '''Write metadata into a file.'''
        self.metadata = {}
        self.metadata['Includes time'] = self.w_time
        self.metadata['w'] = self.w
        self.metadata['s'] = self.s
        self.metadata['Max per class'] = self.max_L
        self.metadata['Min per class'] = self.min_L
        self.metadata['Max points per lc'] = self.max_N
        self.metadata['Min points per lc'] = self.min_N
        self.metadata['Numer of classes'] = self.num_classes
        self.metadata['Classes Info'] = self.splits_metadata

        path = self.save_dir+'metadata_preprocess.json'
        with open(path, 'w') as fp:
            json.dump(self.metadata, fp)
        # Save the light curve parameters for the pandas call
        np.savez(self.save_dir+'lc_parameters',lc_parameters = self.lc_parameters)

    def prepare(self, file_train, save_dir
                , train_size = 0.70, val_size=0.10, test_size=0.2
                , inference= False, inference_folder = None):

        self.set_execution_variables(file_train, save_dir
                        , train_size, val_size, test_size
                        , inference, inference_folder)
        self.parallel_read()
        self.parallel_process()
        # Split only if default split is True
        if self.default_split:
            self.split_train()
        self.get_metadata_split()
        self.serialize_all()
        self.write_metadata_process()

    def prepare_inference(self, save_path=None):
        self.parallel_read()
        self.parallel_process()
        self.serialize_inference(save_path)
