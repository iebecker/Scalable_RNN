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
    def __init__(self, file_train, save_dir='./',
                 max_L=40000, min_L=500, min_N=0, max_N = 2000, w=4, s=2, w_time=True,
                 num_cores=8, train_size = 0.70, val_size=0.10, test_size=0.2,
                 lc_parameters = None, inference= False, inference_folder = None):

        # Set inference mode
        self.inference = inference
        self.inference_folder = inference_folder
        self.save_dir = save_dir

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # We already imposed a minimum number of poitns per light curve
        # Impose a minimum of points per light curve
        if self.inference:
            self.max_L = 1e6 #Arbitrary
            self.min_L = 0
            self.min_N = 0
            self.max_N = 1e6 #Arbitrary
        else:
            self.max_L = max_L
            self.min_L = min_L
            self.max_N = max_N
            self.min_N = min_N

        # Dataset info
        self.file_train = file_train
        # self.file_dev = file_dev
        self.read_datasets()

        # Extract classes and the number of them
        self.classes = list(set(self.data_train.Class))
        self.num_classes = len(self.classes)

        # Container for the data
        self.Labels = []
        self.Matrices = []
        self.IDs = []

        # Parameters for the objects
        self.w = w
        self.s = s
        self.w_time = w_time
        self.njobs = num_cores

        # Dictionary to transform string labels to ints
        self.trans = {c: n for c, n in zip(self.classes, range(self.num_classes))}
        self.trans_inv = dict(zip(self.trans.values(), self.trans.keys()))

        # Parameters to read each light curve
        self.lc_parameters = lc_parameters

        # Set the train/test/val sizes
        self.train_size = train_size

        self.test_size = test_size
        self.val_size = val_size


        # Auxiliary functions to parallelize
        # Parallel does not work with class functions (for some reason)
        self.__func_read = utils.read
        self.__func_process = utils.process
        self.__func_serialize = utils.serialize

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
        Filter the specified number of LC per class, so it does not read everything '''
        # Read stars Data
        self.data_train = pd.read_csv(self.file_train, usecols=['ID', 'Address','Class','N'])

        # Extract classes and the number of them
        self.classes = list(set(self.data_train.Class))
        self.num_classes = len(self.classes)

        # Filter train according to number of observations and elements per class
        self.filter_train()



    def paralell_read(self):
        '''Read the data using n_jobs. Store them in a dict where the classes
        are keys.'''

        # Read the light curves in parallel
        print('Reading')
        ext = Parallel(self.njobs)(delayed(self.__func_read)(address_, class_, id_, self.lc_parameters) for address_, class_, id_ in
                                 tqdm(zip(self.data_train.Address, self.data_train.Class, self.data_train.ID)))

        # Create a dictionary by class
        self.lcs = {c: [] for c in self.classes}
        # For each class, light curve i[0] and the id i[2]
        [self.lcs[i[1]].append([i[0],i[2]]) for i in ext]

    def paralell_process(self):
        '''Extracts the data and transform it into matrix representation.'''
        print('Processing')
        for c in self.classes:
            sel = self.lcs[c]
            # Run the process function in parallel
            self.processed = Parallel(self.njobs)(delayed(self.__func_process)(c, l, self.w, self.s, self.w_time) for l in tqdm(sel))

            # Store in list the information.
            # The order is preserved, so an jth element in all lists will correspond to the same object
            # Change the class to a number, and store it into a list
            [self.Labels.append(self.trans[i[0]]) for i in self.processed]
            # Store the matrix representation into a list
            [self.Matrices.append(i[1].astype(np.float32)) for i in self.processed]
            # Store the IDs into a list
            [self.IDs.append(i[2]) for i in self.processed]

        # Transform the data into matrices
        self.Labels = np.array(self.Labels)
        self.Matrices = np.array(self.Matrices)
        self.IDs = np.array(self.IDs)

        # Shuffle the data
        # Create shuffled indices
        if not self.inference:
            ind = np.arange(len(self.Labels))
            shuffle(ind)
            # Aplly the shuffle dindices
            self.Labels = self.Labels[ind]
            # self.Ns = self.Ns[self.ind]
            self.Matrices = self.Matrices[ind]
            self.IDs = self.IDs[ind]

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
        values_norm = ['({0:.3f} %)'.format(v) for v in values_norm]
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

        splits_labels = [self.Labels_train, self.Labels_val, self.Labels_test]

        values = [self.cls_metadata(labels) for labels in splits_labels]
        keys = ['Train set', 'Train-dev set', 'Dev set', 'Test set']
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

        # with open(self.save_dir+'Train.tfrecord', 'w') as f:
        #     writer = tf.io.TFRecordWriter(f.name)
        #     for i in range(len(self.IDs_train)):
        #         ex = self.__func_serialize(self.Matrices_train[i], self.Labels_train[i],self.IDs_train[i])
        #         writer.write(ex.SerializeToString())

        # with open(self.save_dir+'Val.tfrecord', 'w') as f:
        #     writer = tf.io.TFRecordWriter(f.name)
        #     for i in range(len(self.IDs_val)):
        #         ex = self.__func_serialize(self.Matrices_val[i], self.Labels_val[i], self.IDs_val[i])
        #         writer.write(ex.SerializeToString())
        #
        # with open(self.save_dir+'Test.tfrecord', 'w') as f:
        #     writer = tf.io.TFRecordWriter(f.name)
        #     for i in range(len(self.IDs_test)):
        #         ex = self.__func_serialize(self.Matrices_test[i], self.Labels_test[i], self.IDs_test[i])
        #         writer.write(ex.SerializeToString())


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
        metadata = {}
        metadata['Includes time'] = self.w_time
        metadata['w'] = self.w
        metadata['s'] = self.s
        metadata['Max per class'] = self.max_L
        metadata['Min per class'] = self.min_L
        metadata['Max points per lc'] = self.max_N
        metadata['Min points per lc'] = self.min_N
        metadata['Numer of classes'] = self.num_classes
        metadata['Classes Info'] = self.splits_metadata

        path = self.save_dir+'metadata_preprocess.json'
        with open(path, 'w') as fp:
            json.dump(metadata, fp)
        # Save the light curve parameters for the pandas call
        np.savez(self.save_dir+'lc_parameters',lc_parameters = self.lc_parameters)

    def prepare(self, filename=None):
        self.paralell_read()
        self.paralell_process()
        self.split_train()
        self.serialize_all()
        self.write_metadata_process()

    def prepare_inference(self, save_path=None):
        self.paralell_read()
        self.paralell_process()
        self.serialize_inference(save_path)
