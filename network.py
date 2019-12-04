import json
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix
import tensorflow as tf

class Network():
    def load_preprocess_data(self, path):
        with open(path) as f:
            metadata = json.load(f)
            self.w_time = metadata['Includes time']
            if self.w_time:
                self.window = 2*int(metadata['w'])
            else:
                self.window = int(metadata['w'])
            self.num_classes= metadata['Numer of classes']
            self.size_train = int(metadata['Classes Info']['Train set']['Total'])
            self.size_test = int(metadata['Classes Info']['Test set']['Total'])
            self.size_val = int(metadata['Classes Info']['Val set']['Total'])
            trans = metadata['Classes Info']['Keys']
            keys = [int(k) for k in trans.keys()]
            self.trans = dict(zip(keys, trans.values()))

    def set_train_settings(self, metadata_pre_path, size_hidden=None, rnn_layers=None, fc_units=None, fc_layers=None,
        buffer_size=None, epochs=None, num_cores=None, batch_size=None, dropout=None, lr=None, val_steps=None, max_to_keep=None,
        save_dir='./'):

        self.load_preprocess_data(metadata_pre_path)

        self.size_hidden = size_hidden
        self.rnn_layers = rnn_layers
        self.fc_units = fc_units
        self.fc_layers = fc_layers
        self.buffer_size = buffer_size
        self.epochs = epochs
        self.num_cores = num_cores
        self.batch_size = batch_size
        self.dropout = dropout
        self.lr = lr
        self.val_steps = val_steps
        self.max_to_keep = max_to_keep

        log_folder = save_dir + 'Logs/'
        self.log_folder_train = log_folder + 'train/'
        self.log_folder_val = log_folder + 'val/'

        plot_folder = save_dir + 'Plots/'
        self.plot_folder_train = plot_folder + 'train/'
        self.plot_folder_val = plot_folder + 'val/'


        self.model_dir = save_dir + 'Model/'

        self.save_train_settings(self.model_dir)

    def save_train_settings(self, save_dir):

        if not os.path.exists(save_dir):
            print(save_dir)
            os.makedirs(save_dir)

        metadata = {}
        metadata['size_hidden'] = self.size_hidden
        metadata['rnn_layers'] = self.rnn_layers
        metadata['fc_units'] = self.fc_units
        metadata['fc_layers'] = self.fc_layers
        metadata['buffer_size'] = self.buffer_size
        metadata['epochs'] = self.epochs
        metadata['num_cores'] = self.num_cores
        metadata['batch_size'] = self.batch_size
        metadata['dropout'] = self.dropout
        metadata['lr'] = self.lr
        metadata['val_steps'] = self.val_steps
        metadata['max_to_keep'] = self.max_to_keep
        metadata['window'] = self.window
        metadata['num_classes'] = self.num_classes
        class_keys = self.trans
        keys = [str(k) for k in class_keys.keys()]
        class_keys = dict(zip(keys, class_keys.values()))
        metadata['class_keys'] = class_keys

        path = save_dir+'metadata_train.json'
        with open(path, 'w') as fp:
            json.dump(metadata, fp)

    def load_train_settings(self, metadata_train_path):

        with open(metadata_train_path) as f:
            metadata = json.load(f)
            self.size_hidden = metadata['size_hidden']
            self.rnn_layers = metadata['rnn_layers']
            self.fc_units = metadata['fc_units']
            self.fc_layers = metadata['fc_layers']
            self.buffer_size = metadata['buffer_size']
            self.epochs = metadata['epochs']
            self.num_cores = metadata['num_cores']
            self.batch_size = metadata['batch_size']
            self.dropout = metadata['dropout']
            self.lr = metadata['lr']
            self.val_steps = metadata['val_steps']
            self.max_to_keep = metadata['max_to_keep']
            self.window = metadata['window']
            self.num_classes = metadata['num_classes']
            self.trans = metadata['class_keys']
            keys = [int(k) for k in self.trans.keys()]
            self.trans = dict(zip(keys, self.trans.values()))

    def GRU_cell(self):

        xavier = tf.contrib.layers.xavier_initializer()
        glorot = tf.glorot_normal_initializer()
        gru_cell = tf.contrib.rnn.GRUCell(
            self.size_hidden,
            activation=tf.nn.tanh,
            kernel_initializer=glorot,
            bias_initializer=xavier
            )
        # gru_cell = tf.contrib.rnn.DropoutWrapper(gru_cell, output_keep_prob=1.0 - self.dropout)

        return gru_cell

    def last_relevant(self, output, length):
        batch_size = tf.shape(output)[0]
        max_length = tf.shape(output)[1]
        out_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, out_size])
        relevant = tf.gather(flat, index)

        return relevant

    def add_FC(self, prev_layer, i):
        '''Function to create a fully connected layer.'''
        hidden_layer = tf.contrib.layers.fully_connected(
            inputs=prev_layer,
            num_outputs=self.fc_units)
        dropout_layer = tf.layers.dropout(
            inputs=hidden_layer,
            rate=self.dropout,
            training=self.is_train,
            name='Dropout_'+str(i), noise_shape=None
            )
        return dropout_layer

    def data_parser(self, serialized_example):

        contex_features_ = {
            'N': tf.io.FixedLenFeature([], dtype=tf.int64),
            'ID': tf.io.FixedLenFeature([],dtype=tf.string),
            'Label': tf.io.FixedLenFeature([],dtype=tf.int64),
            }
        sequence_features_={'LightCurve_1': tf.io.VarLenFeature( dtype=tf.float32)}
        context_data, sequence_data = tf.io.parse_single_sequence_example(
            serialized=serialized_example,
            context_features=contex_features_,
            sequence_features=sequence_features_
            )

        N = tf.cast(context_data['N'],tf.int32)
        lbl = tf.cast(context_data['Label'],tf.int32)
        ID_ = tf.cast(context_data['ID'],tf.string)

        lbl = tf.one_hot(lbl, self.num_classes, on_value=1, off_value=0, axis=-1, dtype=tf.int32)
        lc_ = tf.compat.v1.sparse.to_dense(sequence_data['LightCurve_1'])
        lc_ = tf.cast(lc_,tf.float32)
        lc = tf.reshape(lc_, [N, self.window])

        return lc, lbl, N, ID_

    def add_input_iterators(self):
        with tf.device('/cpu:0'), tf.name_scope('Iterators'):
            self.filename_pl = tf.compat.v1.placeholder(tf.string, shape=[None],name='Filename')
            self.epochs_pl = tf.compat.v1.placeholder(tf.int64, shape=[],name='Epochs')
            self.handle_pl = tf.compat.v1.placeholder(tf.string, shape=[],name='Handle')

            dataset = tf.data.TFRecordDataset(self.filename_pl)
            # Repeat epochs_pl times
            dataset = dataset.repeat(count=self.epochs_pl)
            # Deserialize and Parse
            dataset = dataset.map(self.data_parser, num_parallel_calls=self.num_cores)

            self.dataset_train = dataset.shuffle(buffer_size=self.buffer_size)
            self.dataset_train = self.dataset_train.padded_batch(self.batch_size, padded_shapes=([None,None],[self.num_classes],[],[])
                                                ,drop_remainder=False).prefetch(1)
            #Pad the batch lc,lbl,N
            self.dataset_eval = dataset.padded_batch(self.batch_size, padded_shapes=([None,None],[self.num_classes],[],[])
                                    ,drop_remainder=False).prefetch(1)


            output_types = tf.compat.v1.data.get_output_types(self.dataset_train)
            output_shapes = tf.compat.v1.data.get_output_shapes(self.dataset_train)
            self.train_iterator = tf.compat.v1.data.Iterator.from_structure(output_types, output_shapes)
            self.train_initializer = self.train_iterator.make_initializer(self.dataset_train)

            self.eval_iterator = tf.compat.v1.data.make_initializable_iterator(self.dataset_eval)
            self.eval_initializer = self.eval_iterator.initializer


            feedable_iter = tf.compat.v1.data.Iterator.from_string_handle(self.handle_pl, output_types, output_shapes)
            self.next_element = feedable_iter.get_next()

    def add_input_placeholders(self):

        self.data_pl = tf.compat.v1.placeholder(tf.float32, [None, None, self.window],name='Data')
        self.target_pl = tf.compat.v1.placeholder(tf.float32, [None, self.num_classes],name='Label')
        self.length_pl = tf.compat.v1.placeholder(tf.int32, [None],name='Length')
        self.id_pl = tf.compat.v1.placeholder(tf.string, [None],name='ID')
        self.is_train = tf.compat.v1.placeholder(tf.bool, name='IsTrain')
        self.global_step = tf.compat.v1.train.get_or_create_global_step()

    def add_model(self):

        with tf.compat.v1.variable_scope('RNN_'):
            grus = [self.GRU_cell() for _ in range(self.rnn_layers)]
            stacked_gru = tf.contrib.rnn.MultiRNNCell(grus, state_is_tuple=True)
            output, state = tf.nn.dynamic_rnn(stacked_gru, self.data_pl, dtype=tf.float32, sequence_length=self.length_pl, swap_memory=True)

        with tf.compat.v1.variable_scope('Cell_Values'):
            self.last_h = self.last_relevant(output, self.length_pl)
            last_h_hist = tf.compat.v1.summary.histogram('State', self.last_h)

            cell_weights = []
            for w in grus[-1].weights:
                name = w.name.split('/')[-2]+'_'+w.name.split('/')[-1][:-2]
                cell_weights.append(tf.compat.v1.summary.histogram(name, w))

        with tf.compat.v1.variable_scope('FC_'):
            if self.fc_layers>0:
                self.fc = None
                for i in range(self.fc_layers):
                    if self.fc is None:
                        self.fc = self.add_FC(self.last_h, i)
                    else:
                        self.fc = self.add_FC(self.fc, i)
            else:
                self.fc = self.last_h

        with tf.compat.v1.variable_scope('Softmax'):
            if self.fc_layers > 0:
                weight = tf.random.truncated_normal([self.fc_units, self.num_classes], stddev=0.1, dtype=tf.float32)
                weight = tf.Variable(weight, name='Softmax_weight')
            elif self.fc_layers == 0:
                weight = tf.random.truncated_normal([self.size_hidden, self.num_classes], stddev=0.1,dtype=tf.float32)
                weight = tf.Variable(weight, name='SoftMax_weight')
            bias = tf.constant(0.01, shape=[self.num_classes], dtype=tf.float32)
            bias = tf.Variable(bias, name='Softmax_bias')
            logit = tf.matmul(self.fc, weight) + bias

        with tf.compat.v1.variable_scope('Prediction'):
            self.prediction = tf.compat.v1.nn.softmax(logits=logit,name='Softmax_prediction')
            self.pred_index = tf.argmax(self.prediction, axis=1)
            self.targ_index = tf.argmax(self.target_pl, axis=1)
            mistakes = tf.not_equal(self.targ_index, self.pred_index)
            self.err = tf.reduce_mean(tf.cast(mistakes, tf.float32), axis=0)
            err_summ = tf.compat.v1.summary.scalar('Error', self.err)

        with tf.name_scope('Cross_Entropy_Loss'):
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.target_pl, logits=logit, name='Cross_entropy')
            self.loss = tf.reduce_mean(loss)
            loss_summ = tf.compat.v1.summary.scalar('Loss', self.loss)

        with tf.name_scope('Optimizer_'):
            lr = tf.compat.v1.train.exponential_decay(self.lr, self.global_step, 1000, 0.98, staircase=True)
            optimizer = tf.compat.v1.train.AdamOptimizer(lr)
            grads_and_vars = optimizer.compute_gradients(loss, tf.compat.v1.trainable_variables())

            # Gradient clipping
            capped_grads_and_vars = []
            h = []
            for grad, var in grads_and_vars:
                if (grad != None):
                    n = tf.clip_by_value(grad, -5, 5)
                    h.append(tf.compat.v1.summary.histogram('{}_grad_cap'.format(var.name.replace(':','_')), n))
                else:
                    n = grad
                capped_grads_and_vars.append((n, var))

            self.train_step = optimizer.apply_gradients(capped_grads_and_vars, global_step=self.global_step)

        self.summary_op = tf.compat.v1.summary.merge([err_summ, loss_summ, h, last_h_hist, cell_weights])

        self.err_sum_ph = tf.compat.v1.placeholder(tf.float32, shape=[],name='Err_Sum')
        self.loss_sum_ph = tf.compat.v1.placeholder(tf.float32, shape=[],name='Loss_Sum')
        err_sum_ = tf.compat.v1.summary.scalar('Err_', self.err_sum_ph)
        loss_sum_ = tf.compat.v1.summary.scalar('Loss_', self.loss_sum_ph)
        self.summary_op_mod = tf.compat.v1.summary.merge([err_sum_, loss_sum_])

    def add_writers(self):
        graph = tf.compat.v1.get_default_graph()
        self.writer_train = tf.compat.v1.summary.FileWriter(self.log_folder_train, graph, flush_secs=30)
        self.writer_val = tf.compat.v1.summary.FileWriter(self.log_folder_val, graph, flush_secs=30)

    def add_saver(self):
        self.saver = tf.compat.v1.train.Saver(max_to_keep=self.max_to_keep)

    def build_graph(self):
        tf.compat.v1.reset_default_graph()
        self.add_input_placeholders()
        self.add_input_iterators()
        self.add_model()
        self.add_saver()

    def plot_cm_ax(self, ax, cm, normalize):
        if normalize:
            title = 'Normalized Confusion Matrix'
        else:
            title = 'Confusion Matrix'
        ax.set_title(title)

        labels = [self.trans[i] for i in range(self.num_classes)]
        tick_marks = np.arange(self.num_classes)
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(labels, rotation=45)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(labels)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    cm[i,j] ='%.2f' %cm[i,j]

        thresh = 0.001
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i, j] < thresh else 'black')

        ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')


    def plot_cm(self, target, prediction, save_dir, step):

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = save_dir + 'CM-' + str(step) + '.png'

        plt.clf()
        plt.rc('font', size=15)
        plt.rc('axes', titlesize=15)
        plt.rc('axes', labelsize=20)
        plt.rc('xtick', labelsize=15)
        plt.rc('ytick', labelsize=15)
        plt.rc('legend', fontsize=15)
        plt.rc('figure', titlesize=22)

        f, (ax0, ax1) = plt.subplots(1, 2, figsize=(20,12))
        cm = confusion_matrix(target,prediction)
        self.plot_cm_ax(ax0, cm, normalize=False)
        self.plot_cm_ax(ax1, cm, normalize=True)


        cm_files = os.listdir(save_dir)
        cm_files = [os.path.abspath(save_dir + f) for f in cm_files]
        if len(cm_files) >= self.max_to_keep:
            oldest_cm = min(cm_files, key=os.path.getctime)
            os.remove(oldest_cm)

        plt.tight_layout()
        plt.savefig(save_path, format='png', dpi=250)
        plt.close()

    def save_metrics(self, sess, tfrecords, writer, plot_folder, step, name):
        # Initialize eval iterator
        handle = sess.run(self.eval_iterator.string_handle())
        iterator_dict = {self.filename_pl: tfrecords, self.epochs_pl: 1, self.handle_pl: handle}
        sess.run(self.eval_initializer, iterator_dict)

        _preds = np.zeros(0)
        _labels= np.zeros(0)
        _ids = np.zeros(0,dtype=np.int64)
        Loss = 0
        tensors = [ self.loss, self.pred_index, self.targ_index]
        while True:
            try:
                data, labels, lengths, ids = sess.run(self.next_element, feed_dict={self.handle_pl:handle})
                feed_dict = {self.data_pl: data, self.target_pl: labels, self.length_pl: lengths
                , self.id_pl: ids, self.is_train:False}

                loss, predictions, target = sess.run(tensors, feed_dict)

                ### Acumulate results
                Loss += loss*labels.shape[0]
                _preds = np.append(_preds, predictions)
                _labels= np.append(_labels, labels.argmax(1))
                _ids = np.append(_ids, ids)

            except tf.errors.OutOfRangeError:

                bol = _preds!=_labels
                err = np.sum(bol)/bol.shape[0]
                print('Prediction accuracy on {} set: {:3.2f}%'.format(name, 100 * (1-err)))
                # Add the summaries
                summary = sess.run(self.summary_op_mod, feed_dict={self.err_sum_ph: err, self.loss_sum_ph: Loss/bol.shape[0]})
                writer.add_summary(summary, step)
                writer.flush()

                # self.plot_cm(_labels, _preds, plot_folder, step)
                return Loss/bol.shape[0]

    def train(self, train_args, tfrecords_train, tfrecords_val):#, tfrecords_dev):

        tf.compat.v1.reset_default_graph()
        self.set_train_settings(**train_args)
        self.build_graph()
        self.add_writers()

        with tf.compat.v1.Session() as sess:

            self.best_loss = np.inf

            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(tf.compat.v1.local_variables_initializer())

            # Initialize the training iterator
            train_handle = sess.run(self.train_iterator.string_handle())
            iterator_dict = {self.filename_pl: tfrecords_train, self.epochs_pl: self.epochs
                            , self.handle_pl: train_handle}
            sess.run(self.train_initializer, iterator_dict)

            while True:
                try:
                    step = tf.compat.v1.train.global_step(sess, self.global_step)

                    data, labels, lengths, ids = sess.run(self.next_element, feed_dict={self.handle_pl:train_handle})

                    feed_dict = {self.data_pl: data, self.target_pl: labels, self.length_pl: lengths
                                , self.id_pl: ids, self.is_train:True, self.handle_pl:train_handle}
                    sess.run(self.train_step, feed_dict)
                    if step%self.val_steps==0:
                        args_train = [sess, tfrecords_train, self.writer_train, self.plot_folder_train
                        , step, 'Train']
                        args_val = [sess, tfrecords_val, self.writer_val, self.plot_folder_val
                        , step, 'Val']
                        loss_train = self.save_metrics(*args_train)
                        loss_val = self.save_metrics(*args_val)
                        save_path = self.model_dir + 'model.ckpt'
                        self.saver.save(sess, save_path, global_step=step)

                except tf.errors.OutOfRangeError:
                    print('Training ended')
                    self.writer_val.close()
                    self.writer_train.close()
                    break

    def predict(self, tfrecords, model_name, metadata_train_path, return_h=False, return_fc=False, return_p=False):

        tf.compat.v1.reset_default_graph()
        self.load_train_settings(metadata_train_path)
        self.build_graph()

        with tf.compat.v1.Session() as sess:

            self.saver.restore(sess, model_name)

            handle = sess.run(self.eval_iterator.string_handle())
            iterator_dict = {self.filename_pl: tfrecords, self.epochs_pl: 1, self.handle_pl: handle}
            sess.run(self.eval_initializer, iterator_dict)

            _preds = np.zeros(0)
            _labels= np.zeros(0)
            _ids = np.zeros(0,dtype=np.int64)
            _last_h = np.zeros((0,self.size_hidden))
            _fc_out = np.zeros((0,self.fc_units))
            _probs =  np.zeros((0,self.num_classes))
            tensors = [self.last_h, self.fc, self.prediction]

            while True:
                try:
                    data, labels, lengths, ids = sess.run(self.next_element, feed_dict={self.handle_pl:handle})
                    feed_dict = {self.data_pl: data, self.target_pl: labels, self.length_pl: lengths
                    , self.id_pl: ids, self.is_train:False, self.handle_pl:handle}
                    last_h, fc_output, pred_probs = sess.run(tensors, feed_dict)

                    ### Acumulate results
                    _preds = np.append(_preds, pred_probs.argmax(1))
                    _labels= np.append(_labels, labels.argmax(1))
                    _ids = np.append(_ids, ids)
                    _last_h = np.append(_last_h, last_h, axis=0)
                    _fc_out = np.append(_fc_out, fc_output, axis=0)
                    _probs = np.append(_probs, pred_probs, axis=0)
                except tf.errors.OutOfRangeError:
                    # Compute errors
                    bol = _preds!=_labels
                    err = np.sum(bol)/bol.shape[0]
                    labels = np.array([self.trans[i] for i in _labels])
                    pred_label = np.array([self.trans[i] for i in _preds])
                    pred_probs = np.max(_probs, axis=-1)
                    break

            _ids = np.hstack(_ids)

            print(_ids.shape, labels.shape, pred_label.shape, pred_probs.shape, _last_h.shape, _fc_out.shape)
            
            self.predictions = {'ids': _ids, 'labels': labels, 'pred_label': pred_label, 'pred_probs': pred_probs
                            , 'last_h': _last_h, 'fc_output': _fc_out}

            if not return_h:
                self.predictions.pop('last_h')

            if not return_fc:
                self.predictions.pop('fc_output')

            if not return_p:
                self.predictions.pop('pred_probs')

            return self.predictions
