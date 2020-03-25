

```python
from network import Network
from preprocesser import Preprocesser
from time import time
```

This repository contains the code needed to train and evaluate the model presented in [Scalable End-to-end Recurrent Neural Network for Variable star classification
](https://academic.oup.com/mnras/article/493/2/2981/5728517)

All the datasets used are available at this [link](https://drive.google.com/drive/folders/1Ywjz8RKq8fsqQrK3NBiFUVAs1P17y13I?usp=sharing).
Each file contains the light curves compressed and a csv file, wich contains the ID of the object, its class, the number of observations and the relative path ob the light curve of each object.

An example for the Gaia DR2 dataset is shown below.

|    |                  ID | Class      | Path                          |   N |
|----|---------------------|------------|-------------------------------|-----|
| 47 | 5953061724418560000 | MIRA_SR    | ./LCs/5953061724418560000.dat |  25 |
| 49 | 4336314533849303296 | RRAB       | ./LCs/4336314533849303296.dat |  19 |
|  3 | 4041877208419516544 | RRAB       | ./LCs/4041877208419516544.dat |  14 |
|  7 | 1745948461173088128 | RRC        | ./LCs/1745948461173088128.dat |  12 |
| 22 | 6199756429598744832 | DSCT_SXPHE | ./LCs/6199756429598744832.dat |  20 |

To train the model, the data has to be preprocessed first. 

First, we create the Preprocesser object with the following parameters:
1. **max_L**: The maximum number of lightcurves per class.
2. **min_L**: The minimum number of light curves per class. If the number is not enough, the class is discarded.
3. **min_N**: The minimum number of observations of the light curves.
4. **max_N**: The maximum number of observations of the light curves.
5. **w**: The size of the sliding window.
6. **s**: The step of the sliding window.
7. **w_time**: Whether to use the time information of the light curves. 
8. **lc_parameters**: A dictionary containing the parameters given to pandas.read_csv to read each light curve.
9. **num_cores**: Number of threads to use in the preprocessing stage.



```python
max_L = 40000
min_L=500
min_N=20
max_N=2000
w=4
s=2
w_time=True
lc_parameters = {'header':0, 'na_filter':False,'sep':',','usecols':['time', 'mag', 'mag_err']}
num_cores=4
P = Preprocesser(max_L=max_L,
                min_L=min_L,
                min_N=min_N, 
                max_N=max_N, 
                w=w,
                s=s,
                w_time=w_time, 
                lc_parameters=lc_parameters,
                num_cores=num_cores
                )
```

Once the object is created, we preprocess the data with the function **prepare**. 

The parameters needed are:
1. **file_train**: The csv file containing ID, path, class and number of observations, per object.
2. **save_dir**: The path to store the preprocessed files and metadata.
3. **train_size**: the proportion to data to be included in the training set. 
4. **test_size**: the proportion to data to be included in the test set.
5. **val_size**: the proportion to data to be included in the validation set.

If he user wants to add their own splits, **train_size**, **test_size** and **val_size** must have the same structure as **file_train**.


```python
t_ini = time()

file_train = './GAIA_dataset.dat'
save_dir = './Output/'
train_size = 0.7
test_size = 0.2
val_size = 0.1
P.prepare(file_train=file_train, save_dir=save_dir, train_size=train_size, test_size=test_size, val_size=val_size)

t_end = time()
print('Time elapsed: {:2.2f} minutes.'.format((t_end-t_ini)/60))

```

    0it [00:00, ?it/s]

    Reading


    114442it [01:08, 1666.97it/s]
     30%|██▉       | 6234/21116 [00:00<00:00, 46760.40it/s]

    Processing


    100%|██████████| 21116/21116 [00:00<00:00, 37066.38it/s]
    100%|██████████| 40000/40000 [00:01<00:00, 27412.86it/s]
    100%|██████████| 6274/6274 [00:00<00:00, 58820.91it/s]
    100%|██████████| 1308/1308 [00:00<00:00, 24698.59it/s]
    100%|██████████| 5178/5178 [00:00<00:00, 62557.73it/s]
    100%|██████████| 40000/40000 [00:01<00:00, 32971.36it/s]
    100%|██████████| 566/566 [00:00<00:00, 51979.94it/s]


    Time elapsed: 1.50 minutes.


Once preprocessed, the folder **save_dir** will contain the serialized files **Train.tfrecord**, **Test.tfrecord** and **Val.tfrecord**.
Additionally, it will store **metadata_preprocess.json** containing the metadata of the preocess and a numpy serialized file containing **lc_parameters**.

To preprocess another dataset without validation splits, the Preprocesser object has the function **prepare_inference**.
The parameters needed are:
1. **file_train**: The csv file containing ID, path, class and number of observations, per object.
2. **save_dir**: The path to store the preprocessed files and metadata.
3. **metadata_path**: The path of the _metadata_preprocess.json_ file.



```python
file_train = './GAIA_dataset_inference.dat'
save_path = './Output/Inference.tfrecord'
metadata_path = './Output/metadata_preprocess.json'
P.prepare_inference(file_train=file_train
                    , save_path=save_path
                    , metadata_path=metadata_path)
```

    50it [00:00, 7936.54it/s]
    100%|██████████| 8/8 [00:00<00:00, 953.31it/s]
    100%|██████████| 24/24 [00:00<00:00, 7330.03it/s]
    100%|██████████| 1/1 [00:00<00:00, 1207.34it/s]
    100%|██████████| 16/16 [00:00<00:00, 4220.68it/s]
    100%|██████████| 1/1 [00:00<00:00, 1313.59it/s]

    Reading
    Processing


    


To train a model, we create the object Network without any parameters.

The function _train_ receives the dictionary **train_args**, and the serialized paths **tfrecords_trainfor** and **tfrecords_val**, to the training and validation files, respectively.

It contains the following keys:
1. **epochs**: The number of epochs used to train the model.
2. **size_hidden**: Size of the hidden state.
3. **rnn_layers**: Number of recurrent layers.
4. **lr**: learning rate.
5. **fc_layers**: Number of fully connected layers to be applied after the recurrent portion
6. **fc_units**: Size of the fully connected layer. By default is the double of the hidden state size.
7. **batch_size**: Size of the batch in the training stage. 
8. **dropout**: Percentaje of dropout used in the fully connected layers. Default 0.
9. **val_steps**: Number of training steps before evaluating in the validation set.
10. **num_cores**: Number of cores used to deserialize the information and feed the GPU. 
11. **buffer_size**: Size of the buffer which shuffles the data.
12. **max_to_keep**: Maximum number of models to keep.
13. **metadata_pre_path**: Path of _metadata_preprocess.json_ file.
14. **buffer_size**: Size of the shuffle buffer.
15. **num_cores**: Number of threads to use in the input pipeline.
16. **save_dir**: Path to save the training data.

In the path specified in **save_dir**, the folders _Model_ to store the model checkpoints and _Logs_ which can be visualized in tensorboard.


```python
t_ini = time()
data_dir = './Output/'
tfrecords_train = [data_dir+'Train.tfrecord']
tfrecords_val = [data_dir+'Val.tfrecord']


train_args = {
'size_hidden' : 25,
'rnn_layers' : 2,
'fc_units' : 50,
'fc_layers' : 1,
'buffer_size' : 40000,
'epochs' : 5,
'num_cores' : 7,
'batch_size' : 2500,
'dropout' : 0.4,
'lr' : 1e-2,
'val_steps' : 1,
'max_to_keep' : 0,
'metadata_pre_path': data_dir+'metadata_preprocess.json',
'save_dir' : './Results/'
}

net = Network()
net.train(train_args, tfrecords_train, tfrecords_val)
t_end = time()
print('Time elapsed: {:2.2f} minutes.'.format((t_end-t_ini)/60))
```

    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/data/ops/iterator_ops.py:347: Iterator.output_types (from tensorflow.python.data.ops.iterator_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.compat.v1.data.get_output_types(iterator)`.
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/data/ops/iterator_ops.py:348: Iterator.output_shapes (from tensorflow.python.data.ops.iterator_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.compat.v1.data.get_output_shapes(iterator)`.
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/data/ops/iterator_ops.py:350: Iterator.output_classes (from tensorflow.python.data.ops.iterator_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.compat.v1.data.get_output_classes(iterator)`.
    WARNING:tensorflow:
    The TensorFlow contrib module will not be included in TensorFlow 2.0.
    For more information, please see:
      * https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md
      * https://github.com/tensorflow/addons
      * https://github.com/tensorflow/io (for I/O related ops)
    If you depend on functionality not listed there, please file an issue.
    
    WARNING:tensorflow:From /home/chispa/Dropbox/PhD/2019/2/Papers/Paper_1/Published_Code/network.py:111: GRUCell.__init__ (from tensorflow.python.ops.rnn_cell_impl) is deprecated and will be removed in a future version.
    Instructions for updating:
    This class is equivalent as tf.keras.layers.GRUCell, and will be replaced by that in Tensorflow 2.0.
    WARNING:tensorflow:From /home/chispa/Dropbox/PhD/2019/2/Papers/Paper_1/Published_Code/network.py:208: MultiRNNCell.__init__ (from tensorflow.python.ops.rnn_cell_impl) is deprecated and will be removed in a future version.
    Instructions for updating:
    This class is equivalent as tf.keras.layers.StackedRNNCells, and will be replaced by that in Tensorflow 2.0.
    WARNING:tensorflow:From /home/chispa/Dropbox/PhD/2019/2/Papers/Paper_1/Published_Code/network.py:209: dynamic_rnn (from tensorflow.python.ops.rnn) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `keras.layers.RNN(cell)`, which is equivalent to this API
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/ops/rnn_cell_impl.py:559: Layer.add_variable (from tensorflow.python.keras.engine.base_layer) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `layer.add_weight` method instead.
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/ops/rnn.py:244: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.where in 2.0, which has the same broadcast rule as np.where
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow_core/contrib/layers/python/layers/layers.py:1866: Layer.apply (from tensorflow.python.keras.engine.base_layer) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `layer.__call__` method instead.
    WARNING:tensorflow:From /home/chispa/Dropbox/PhD/2019/2/Papers/Paper_1/Published_Code/network.py:134: dropout (from tensorflow.python.layers.core) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use keras.layers.dropout instead.
    Prediction accuracy on Train set: 32.40%
    Prediction accuracy on Val set: 31.37%
    INFO:tensorflow:./Results/Model/model.ckpt-0 is not in all_model_checkpoint_paths. Manually adding it.
    Training ended
    Time elapsed: 0.31 minutes.


Once the training ends, we can predict using the predict method. 

The parameters are:
1. **tfrecords**: List of tfrecord files. The results will be concatenated.
2. **model_name**: Path to the model files identified with the model number. Example, './Model/model.ckpt-0'.
3. **metadata_train_path**: Path to the train metadata, located at './Model/metadata_train.json'.
4. **return_h**: Boolean. Wether to return the hidden state after the RNN section.
5. **return_p**: Boolean. Wether to return the classification probability.

It returns a dictionary with keys _ids_, _labels_, _pred_label_. If selected,  _pred_probs_ and _last_h_.



```python
tfrecords_test = [data_dir+'Test.tfrecord', data_dir+'Inference.tfrecord']
model_name = './Results/Model/model.ckpt-0'
metadata_train_path = './Results/Model/metadata_train.json'

net = Network()
predictions = net.predict(tfrecords_test, model_name, metadata_train_path, return_h=False, return_p=True)
```

    INFO:tensorflow:Restoring parameters from ./Results/Model/model.ckpt-0



```python
predictions = pd.DataFrame(predictions)
predictions.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ids</th>
      <th>labels</th>
      <th>pred_label</th>
      <th>pred_probs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5385985372548272128</td>
      <td>RRAB</td>
      <td>RRAB</td>
      <td>0.191029</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6248180369093869440</td>
      <td>RRC</td>
      <td>RRAB</td>
      <td>0.191314</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5951956131140637440</td>
      <td>RRC</td>
      <td>RRAB</td>
      <td>0.224031</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5968726840210212352</td>
      <td>MIRA_SR</td>
      <td>RRAB</td>
      <td>0.207551</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5927671080359624064</td>
      <td>MIRA_SR</td>
      <td>RRAB</td>
      <td>0.198111</td>
    </tr>
  </tbody>
</table>
</div>


