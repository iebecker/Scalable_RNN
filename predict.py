import os
import sys
from network import Network
import tensorflow as tf

tfrecords = '../Runs/Serialized/Xmatch.tfrecord'
model_name = '../Runs/job1/Model/model.ckpt-26500'
metadata_train_path = '../Runs/job1/Model/metadata_train.json'

net = Network()
predictions = net.predict(tfrecords, model_name, metadata_train_path, return_h=False, return_fc=False, return_p=False)