import os
import preprocesser as p


data_dir  = './Databases/Gaia/data_2/G/'
data_save = './Serialized/Saved_Data/'
file_train = './Splits/Train.dat'
file_test = './Splits/Test.dat'
file_val = './Splits/Val.dat'


min_L = 100
max_L = 50
min_N = 15
max_N = 2000
w = 4
s = 2
time = False

lc_parameters = {'header':0, 'na_filter':False,'sep':',','usecols':['time', 'mag', 'mag_err']}
num_cores = 7

preprocesser = Prep.Preprocesser(max_L=max_L, min_L=min_L, min_N=min_N, max_N=max_N, w=w, s=s,
               num_cores=num_cores, lc_parameters=lc_parameters,w_time = time)

# Excecute the inference
preprocesser.prepare(file_train=file_train, save_dir=data_save, train_size=file_train, test_size=file_test, val_size = file_val)