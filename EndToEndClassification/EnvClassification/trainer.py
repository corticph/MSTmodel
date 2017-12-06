import tensorflow as tf
import numpy as np
import os

from EndToEndClassification.Utilities import dump_pickle, classification_accuracy


class ClassifierTrainer():
    """
    Trainer for an environmental sound classifier. Will infer the correct folds for saving from the dataset as well 
    as whether it's being trained on raw speech or log Mel-spectrograms from the input.
    
    To use the trainer with the first 3 layers pretrained to the log Mel-spectrogram transformation provide a 
    restore path to the 'MSTmodel_initialized' parameter.
    """

    def __init__(self, model, dataset, save_folder, seed=42, save_model=True,
                 MSTmodel_initialized=False, piczak_initialized=False, MSTfrozen=True, Piczakfrozen=False,
                 save_separate=True):
        """
        Initializes the trainer class.
        
        Args:
            model (class): initialized model (Piczak or RawPiczak).
            dataset (class): loaded (ESC50) dataset (ClassifierLoader).
            save_folder (str): save folder where the results (and optionally the model) will be saved.
            seed (int): seed for initializing the pseudo-rng for reproducibility purposes.
            save_model (bool): if True the model is saved as well as the results.
            MSTmodel_initialized (bool or str): if False the raw speech model is trained with random initialization
                                            else a string must be provided to a trained MSTmodel and the first 3 layers 
                                            are initialized with the parameters from that model.
            piczak_initialized (bool or str): optionally initialize the Piczak part of the model with one that was
                                            previously trained.
            MSTfrozen (bool): whether to keep the MSTmodel parameters fixed (leave to True to reproduce results)
            Piczakfrozen (bool) whether to keep the Piczak parameters fixed (leave to False to reproduce results)
            save_separate (bool): whether to make separate subfolders for the MSTmodel and piczak parts of the
                                            network and save the parameters separately, or to save the model as
                                            a whole.
        """

        self._rng = np.random.RandomState(seed=seed)
        self.model = model
        self.dataset = dataset

        results_filename = 'results' + '_' + str(dataset.validation_fold) + str(dataset.test_fold) + '.pkl'

        if not os.path.isdir(save_folder):
            raise ValueError('please provide a valid save folder')

        self.save_results = os.path.join(save_folder, results_filename)

        self.save_separate = save_separate

        if save_model:
            self.save_model = True
            model_name = self.model.model_name + '_' + str(dataset.validation_fold) + str(dataset.test_fold)

            if self.save_separate:

                save_model_path = os.path.join(save_folder, model_name)
                os.mkdir(save_model_path)
                self.save_model_path_mst = os.path.join(save_model_path, 'MST_' + model_name)
                os.mkdir(self.save_model_path_mst)
                self.save_model_path_piczak = os.path.join(save_model_path, 'piczak_' + model_name)
                os.mkdir(self.save_model_path_piczak)
            else:
                self.save_model_path = os.path.join(save_folder, model_name)
                os.mkdir(self.save_model_path)

        if MSTmodel_initialized is not False:
            if not isinstance(MSTmodel_initialized, str):
                raise ValueError('please indicate training from scratch or provide a path to a pretrained MSTmodel')

        self.MSTmodel_initialized = MSTmodel_initialized
        self.piczak_initialized = piczak_initialized
        self.MSTfrozen = MSTfrozen
        self.Piczakfrozen = Piczakfrozen

        # the attributes below are set by the train() method
        self.no_epochs = None
        self.batch_size = None
        self.lr = None
        self.momentum = None

        self.input_batch_shape = None
        self.label_batch_shape = None
        self.eval_input_shape = None
        self.eval_label_shape = None
        self.test_input_shape = None
        self.test_label_shape = None

        self.input_placeholder = None
        self.label_placeholder = None
        self.evaluation_input_placeholder = None
        self.evaluation_label_placeholder = None
        self.test_input_placeholder = None
        self.test_label_placeholder = None

        self.train_prediction_op = None
        self.eval_prediction_op = None
        self.test_prediction_op = None
        self.train_loss_op = None
        self.val_op = None
        self.test_op = None

        self.grad_op = None
        self.train_op = None
        self.sess = None

    def train(self, batch_size=500, no_epochs=200, lr=5e-3, momentum=0.9, momentum_optimizer=True):
        """
        Performs training of the classifier with a pre-defined number of epochs and a constant learning rate. 
        The momentum optimizer is used by default, else Adam.
        
        Args:
            batch_size (int): number of train examples in a mini-batch.
            no_epochs (int): number of epochs (pre-defined, no overfitting test implemented in this set-up).
            lr (float): constant learning rate.
            momentum (float): momentum value of the optimizer.
            momentum_optimizer (bool): whether to use the momentum optimizer (True) or Adam (False). Leave to True
                                        to reproduce the paper results.
        """

        self.no_epochs = no_epochs
        self.batch_size = batch_size

        # check whether training is done on raw speech or on log-Mel spectrograms
        if len(self.dataset.train[0].shape) == 4:
            self.input_batch_shape = (batch_size, self.dataset.train[0].shape[1], self.dataset.train[0].shape[2],
                                      self.dataset.train[0].shape[3])
        elif len(self.dataset.train[0].shape) == 3:
            self.input_batch_shape = (batch_size, self.dataset.train[0].shape[1], self.dataset.train[0].shape[2])
        else:
            raise ValueError('Incorrect input data dimensionality')

        # retrieve the shapes for the placeholders from the dataset
        self.label_batch_shape = (batch_size)
        self.eval_input_shape = self.dataset.validation[0].shape
        self.eval_label_shape = self.dataset.validation[1].shape
        self.test_input_shape = self.dataset.test[0].shape
        self.test_label_shape = self.dataset.test[1].shape

        # optimizer related
        self.lr = lr
        self.momentum = momentum

        # intialize the loss
        self.best_loss = np.inf

        # build the graph
        self.input_placeholder = tf.placeholder(tf.float32, shape=self.input_batch_shape)
        self.label_placeholder = tf.placeholder(tf.int32, shape=self.label_batch_shape)

        self.evaluation_input_placeholder = tf.placeholder(tf.float32, shape=self.eval_input_shape)
        self.evaluation_label_placeholder = tf.placeholder(tf.int32, shape=self.eval_label_shape)

        self.test_input_placeholder = tf.placeholder(tf.float32, shape=self.test_input_shape)
        self.test_label_placeholder = tf.placeholder(tf.int32, shape=self.test_label_shape)

        self.is_training_ph = tf.placeholder(tf.bool, name='is_training')

        with tf.variable_scope("model"):
            self.train_prediction_op = self.model.build_predict_op(input_tensor=self.input_placeholder,
                                                                   is_training=self.is_training_ph)
        with tf.variable_scope("model", reuse=True):
            self.eval_prediction_op = self.model.build_predict_op(input_tensor=self.evaluation_input_placeholder,
                                                                  is_training=self.is_training_ph)
            self.val_op = tf.argmax(self.eval_prediction_op, axis=1)
            self.test_prediction_op = self.model.build_predict_op(input_tensor=self.test_input_placeholder,
                                                                  is_training=self.is_training_ph)
            self.test_op = tf.argmax(self.test_prediction_op, axis=1)

        self.train_loss_op = self.model.get_loss_op(prediction=self.train_prediction_op,
                                                    label_tensor=self.label_placeholder)
        with tf.variable_scope("optimizer"):
            if momentum_optimizer:
                optimize = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=self.momentum, use_nesterov=True)
            else:
                optimize = tf.train.AdamOptimizer(learning_rate=self.lr)

            if self.MSTfrozen:
                # only compute gradients with respect to the PiczakCNN variables
                var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "model/piczak")
                self.grad_op = optimize.compute_gradients(loss=self.train_loss_op, var_list=var_list)
            elif self.Piczakfrozen:
                # only compute gradients with respect to the mst variables
                var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "model/MSTmodel")
                self.grad_op = optimize.compute_gradients(loss=self.train_loss_op, var_list=var_list)
            else:
                self.grad_op = optimize.compute_gradients(loss=self.train_loss_op)

            self.train_op = optimize.apply_gradients(self.grad_op)

        # initialize the variables/session
        init = tf.global_variables_initializer()
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(init)

        if self.MSTmodel_initialized is not False:
            self.model.load_MSTmodel(self.MSTmodel_initialized, self.sess)

        if self.piczak_initialized is not False:
            self.model.load_piczak(self.piczak_initialized, self.sess)

        # initialize results storage dict
        results = {
            'train_losses': [],
            'val_accuracy': [],
            'test_accuracy': []
        }

        epoch_train_losses = []
        epoch_val_accuracy = []
        epoch_test_accuracy = []

        for epoch in range(self.no_epochs):

            train_batch_losses = []
            indices_train_batches = self.dataset.make_batch_indices(batch_size)

            # train loop
            for i, batch_indices in enumerate(indices_train_batches):
                input_batch, label_batch = self.dataset.load_batch(batch_indices)

                # the train loss is retrieved with is_training=True which is correct for computing the gradients but
                # not for reporting. However, we're not really interested in the exact train loss and have chosen
                # to save out on a extra forward pass
                _, loss = self.sess.run([self.train_op, self.train_loss_op],
                                        feed_dict={self.input_placeholder: input_batch,
                                                   self.label_placeholder: label_batch,
                                                   self.is_training_ph: True})
                train_batch_losses.append(loss)

            epoch_train_average = np.average(train_batch_losses)

            print('train loss: {}'.format(epoch_train_average))
            epoch_train_losses.append(epoch_train_average)

            results['train_losses'] = epoch_train_losses

            # now retrieve the validation and test accuracy
            val_input, val_labels = self.dataset.validation

            predicted_labels_val = self.sess.run([self.val_op],
                                                 feed_dict={self.evaluation_input_placeholder: val_input,
                                                            self.evaluation_label_placeholder: val_labels,
                                                            self.is_training_ph: False})
            predicted_labels_val = predicted_labels_val[0]

            val_accuracy = classification_accuracy(self.dataset.validation_pd, predicted_labels_val)
            epoch_val_accuracy.append(val_accuracy)

            results['val_accuracy'] = epoch_val_accuracy
            print('val accuracy: {}'.format(val_accuracy))

            test_input, test_labels = self.dataset.test

            predicted_labels_test = self.sess.run([self.test_op],
                                                  feed_dict={self.test_input_placeholder: test_input,
                                                             self.test_label_placeholder: test_labels,
                                                             self.is_training_ph: False})

            predicted_labels_test = predicted_labels_test[0]
            test_accuracy = classification_accuracy(self.dataset.test_pd, predicted_labels_test)
            epoch_test_accuracy.append(test_accuracy)

            results['test_accuracy'] = epoch_test_accuracy
            print('test accuracy: {}'.format(test_accuracy))

            # overwrite the results each time an epoch is done
            dump_pickle(self.save_results, results)

            # if model saving is set to True also overwrite the model after each epoch
            if self.save_model:
                if self.save_separate:
                    self.model.save_MSTmodel(os.path.join(self.save_model_path_mst, self.model.model_name + '_MST'),
                                             self.sess)
                    self.model.save_piczak(os.path.join(self.save_model_path_piczak, self.model.model_name + '_piczak'),
                                           self.sess)
                else:
                    self.model.save(os.path.join(self.save_model_path, self.model.model_name), self.sess)
