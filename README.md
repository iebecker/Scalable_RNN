# README

This repository contains the code needed to train and evaluate the model presented in [Scalable End-to-end Recurrent Neural Network for Variable star classification
](https://academic.oup.com/mnras/article/493/2/2981/5728517?guestAccessKey=9d179a55-61c6-4729-a583-b345fdefa2c2)

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


Once the object is created, we preprocess the data with the function **prepare**. The parameters needed are:
1. **file_train**: The csv file containing ID, path, class and number of observations, per object.
2. **save_dir**: The path to store the preprocessed files and metadata.
3. **train_size**: the proportion to data to be included in the training set. 
4. **test_size**: the proportion to data to be included in the test set.
5. **val_size**: the proportion to data to be included in the validation set.

If he user wants to add their own splits, _train_size_, _test_size_ and _val_size_ must have the same structure as _file_train_.

Once preprocessed, the folder _save_dir_ will contain the serialized files _Train.tfrecord_, _Test.tfrecord_ and _Val.tfrecord_. 
Additionally, it will store _metadata_preprocess.json_ containing the metadata of the preocess and a numpy serialized file containing _lc_parameters_.

To preprocess another dataset without validation splits, the Preprocesser object has the function **prepare_inference**.
The parameters needed are:
1. **file_train**: The csv file containing ID, path, class and number of observations, per object.
2. **save_dir**: The path to store the preprocessed files and metadata.
3. **metadata_path**: The path of the _metadata_preprocess.json_ file.


To train a model, we create the object Network without any parameters.

Then, the function train receives **train_args**, and the serialized files for the training and validation.
**train_args** is a dictionary that contains the following keys:
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

Logs will be stored at save_dir + 'Logs/'.

Once the training ends, you can predict using the predict method. 
The parameters are:
1. **tfrecords**: List of tfrecord files. The results will be concatenated.
2. **model_name**: Path to the model files identified with the model number. Example, './Model/model.ckpt-8000'.
3. **metadata_train_path**: Path to the train metadata, located at './Model/metadata_train.json'.
4. **return_h**: Boolean. Wether to return the hidden state after the RNN section.
5. **return_p**: Wether to return the classification probability.

It returns a dictionary with keys _ids_, _labels_, _pred_label_. If selected,  _pred_probs_ and _last_h_.



