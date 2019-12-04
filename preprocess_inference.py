import os
import preprocesser as p

save_dir = '../Runs/Serialized/'
save_path = '../Runs/Serialized/Xmatch.tfrecords'
file_train = '/Users/Javiera/Desktop/Harvard/Code/Data/features/ts-model/datasets/preprocess/Gaia_xmatch.dat'

w = 4
s = 2
time = False

lc_parameters = {'header':0, 'na_filter':False,'sep':',','usecols':['time', 'mag', 'mag_err']}
trans =  {'RRC': 0, 'RRAB': 1, 'MIRA_SR': 2, 'RRD': 3, 'DSCT_SXPHE': 4}
num_cores = 4

preprocesser = p.Preprocesser(w=w, s=s, num_cores=num_cores, lc_parameters=lc_parameters, w_time = time)

# Excecute the inference
preprocesser.prepare_inference(file_train=file_train, save_dir=save_dir, save_path=save_path, trans=trans)