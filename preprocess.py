import os
import preprocesser as p

data_dir  = '/Users/Javiera/Desktop/Harvard/Code/Data/features/ts-model/datasets/'
data_save = data_dir + 'train/all/'
file_train = data_dir + 'preprocess/Gaia_xmatch.dat'
file_dev = data_dir + 'preprocess/Gaia_xmatch.dat'

min_L = 500
max_L = 5000
min_N = 15
max_N = 2000
w = 4
s = 2
lc_parameters = {'header':0, 'na_filter':False,'sep':',','usecols':['time', 'mag', 'mag_err']}
num_cores = 4

preprocesser = p.Preprocesser(file_train=file_train, file_dev=file_dev, save_dir='../datasets/train/all/',
               max_L=max_L, min_L=min_L, min_N=min_N, max_N=max_N, w=w, s=s,
               num_cores=num_cores, lc_parameters=lc_parameters)

# Excecute the inference
preprocesser.prepare()
