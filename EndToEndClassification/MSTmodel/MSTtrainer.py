import tensorflow as tf
import numpy as np
import os

from EndToEndClassification.Utilities import dump_pickle


class MSTtrainer():
    """
    Trainer class for learning the logscaled melspectrogram transformation.
    """

    def __init__(self, model, dataset, save_folder, restore_path=None, seed=42):
        """
        Initializes the trainer. 
        
        Args:
            model (class): initialized model (MSTmodel).
            dataset (class): loaded (ESC50) dataset (MSTLoader).
            save_folder (str): save folder where the results, the model, and the predictions will be saved.
            restore_path (str or None): random initialization if None, else provide path to trained model.
            seed (int): seed for initializing the pseudo-rng for reproducibility purposes.
        """

        self._rng = np.random.RandomState(seed=seed)
        self.model = model
        self.dataset = dataset
        self.restore_path = restore_path

        results_filename = 'MSTresults' + '_' + str(dataset.validation_fold) + str(dataset.test_fold) + '.pkl'
        predictions_filename = 'MSTpredictions' + '_' + str(dataset.validation_fold) + str(dataset.test_fold) + '.pkl'

        if not os.path.isdir(save_folder):
            raise ValueError('please provide a valid save folder')

        self.save_results = os.path.join(save_folder, results_filename)
        self.save_predictions = os.path.join(save_folder, predictions_filename)

        model_name = self.model.model_name + '_' + str(dataset.validation_fold) + str(dataset.test_fold)
        self.save_model_path = os.path.join(save_folder, model_name)
        os.mkdir(self.save_model_path)

        # the attributes below are set by the train() method
        self.no_epochs = None
        self.batch_size = None
        self.lr = None
        self.overfit_window = None

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
        self.eval_loss_op = None
        self.test_loss_op = None

        self.grad_op = None
        self.train_op = None
        self.sess = None

    def train(self, batch_size=100, no_epochs=800, lr=3e-4, overfit_window=50):
        """
        Performs training of the MSTmodel with early stopping based on the validation loss.
        The Adam optimizer with constant learning rate is used. 
        
        Args:
            batch_size (int): number of train examples in a mini-batch.
            no_epochs (int): number of epochs (pre-defined, no overfitting test implemented in this set-up).
            lr (float): constant learning rate.
            overfit_window (int): if the validation loss has not improved for >overfit_window number of epochs
                                training is stopped.
        """
        self.no_epochs = no_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.overfit_window = overfit_window
        self.best_loss = np.inf

        self.input_batch_shape = (batch_size, self.dataset.train[0].shape[1], self.dataset.train[0].shape[2])
        self.label_batch_shape = (batch_size, self.dataset.train[1].shape[1], self.dataset.train[1].shape[2] - 1,
                                  self.dataset.train[1].shape[3])

        self.eval_input_shape = (self.dataset.validation[0].shape[0], self.dataset.validation[0].shape[1],
                                 self.dataset.validation[0].shape[2])
        self.eval_label_shape = (self.dataset.validation[1].shape[0], self.dataset.validation[1].shape[1],
                                 self.dataset.validation[1].shape[2] - 1, self.dataset.validation[1].shape[3])

        self.test_input_shape = (self.dataset.test[0].shape[0], self.dataset.test[0].shape[1],
                                 self.dataset.test[0].shape[2])
        self.test_label_shape = (self.dataset.test[1].shape[0], self.dataset.test[1].shape[1],
                                 self.dataset.test[1].shape[2] - 1, self.dataset.test[1].shape[3])

        # build the graph
        self.input_placeholder = tf.placeholder(tf.float32, shape=self.input_batch_shape)
        self.label_placeholder = tf.placeholder(tf.float32, shape=self.label_batch_shape)

        self.evaluation_input_placeholder = tf.placeholder(tf.float32, shape=self.eval_input_shape)
        self.evaluation_label_placeholder = tf.placeholder(tf.float32, shape=self.eval_label_shape)

        self.test_input_placeholder = tf.placeholder(tf.float32, shape=self.test_input_shape)
        self.test_label_placeholder = tf.placeholder(tf.float32, shape=self.test_label_shape)

        with tf.variable_scope("model"):
            self.train_prediction_op = self.model.build_predict_op(input_tensor=self.input_placeholder,
                                                                   is_training=True)
        with tf.variable_scope("model", reuse=True):
            self.eval_prediction_op = self.model.build_predict_op(input_tensor=self.evaluation_input_placeholder,
                                                                  is_training=False)
            self.test_prediction_op = self.model.build_predict_op(input_tensor=self.test_input_placeholder,
                                                                  is_training=False)

        self.train_loss_op = self.model.get_loss_op(predictions=self.train_prediction_op,
                                                    labels=self.label_placeholder)
        self.eval_loss_op = self.model.get_loss_op(predictions=self.eval_prediction_op,
                                                   labels=self.evaluation_label_placeholder)
        self.test_loss_op = self.model.get_loss_op(predictions=self.test_prediction_op,
                                                   labels=self.test_label_placeholder)

        rate = self.lr

        with tf.variable_scope("optimizer"):
            optimize = tf.train.AdamOptimizer(learning_rate=rate)

            self.grad_op = optimize.compute_gradients(loss=self.train_loss_op)
            self.train_op = optimize.apply_gradients(self.grad_op)

        # initialize the variables/session
        init = tf.global_variables_initializer()
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        self.sess = tf.Session(config=config)
        self.sess.run(init)

        # restore model parameters if provided
        if self.restore_path is not None:
            self.model.load(self.restore_path, sess=self.sess)

        # initialize storage for saving the results
        results = {}
        results['train_losses'] = []
        results['val_losses'] = []
        results['test_losses'] = []

        # initialize storage for saving predictions and labels of 10 log Mel-spectrograms
        predictions = {}
        predictions['predictions_train'] = None
        predictions['labels_train'] = None
        predictions['predictions_val'] = None
        predictions['labels_val'] = None
        predictions['predictions_test'] = None
        predictions['labels_test'] = None

        epoch_train_losses = []
        epoch_val_losses = []
        epoch_test_losses = []

        overfit_counter = 0

        for epoch in range(self.no_epochs):

            train_batch_losses = []

            indices_train_batches = self.dataset.make_batch_indices(batch_size)

            # train loop
            for i, batch_indices in enumerate(indices_train_batches):
                input_batch, label_batch = self.dataset.load_batch(batch_indices)

                # slice of the last time step for the labels to match dimensions with the predictions
                label_batch = label_batch[:, :, :-1, :]

                # Note: this loss is correct for computing gradients, but if 'is training' layers are used it is not
                # the correct train loss to document
                _, loss, predicted_label_batch = self.sess.run([self.train_op, self.train_loss_op,
                                                                self.train_prediction_op],
                                                               feed_dict={self.input_placeholder: input_batch,
                                                                          self.label_placeholder: label_batch})
                train_batch_losses.append(loss)

                # for starters, just only save the first batch, and overwrite each epoch
                # only save if there's improvement in terms of val loss
                if i == 0:
                    train_labels_for_epoch = label_batch
                    train_predictions_for_epoch = predicted_label_batch

            epoch_train_average = np.average(train_batch_losses)

            print('train loss: {}'.format(epoch_train_average))
            epoch_train_losses.append(epoch_train_average)

            # now retrieve val and test loss and predictions
            val_input, val_labels = self.dataset.validation

            # slice of the last time step for the labels to match dimensions with the predictions
            val_labels = val_labels[:, :, :-1, :]

            val_loss, val_predictions = self.sess.run([self.eval_loss_op, self.eval_prediction_op],
                                                      feed_dict={self.evaluation_input_placeholder: val_input,
                                                                 self.evaluation_label_placeholder: val_labels})
            print('val loss: {}'.format(val_loss))

            test_input, test_labels = self.dataset.test

            # slice of the last time step for the labels to match dimensions with the predictions
            test_labels = test_labels[:, :, :-1, :]

            test_loss, test_predictions = self.sess.run([self.test_loss_op, self.test_prediction_op],
                                                        feed_dict={self.test_input_placeholder: test_input,
                                                                   self.test_label_placeholder: test_labels})
            print('test loss: {}'.format(test_loss))

            epoch_val_losses.append(val_loss)
            epoch_test_losses.append(test_loss)

            results['train_losses'] = epoch_train_losses
            results['val_losses'] = epoch_val_losses
            results['test_losses'] = epoch_test_losses

            # overwrite the results each time an epoch is done
            dump_pickle(self.save_results, results)

            # now test if the loss improved and only then overwrite the model and the predictions
            if val_loss <= self.best_loss:
                print('saving model and predictions since val_loss: {} < {}: best loss before'.format(val_loss,
                                                                                                      self.best_loss))
                self.model.save(os.path.join(self.save_model_path, self.model.model_name), self.sess)

                # save only the first 10 samples for each of the datasets
                num_samples = 10
                predictions['predictions_train'] = train_predictions_for_epoch[:num_samples]
                predictions['labels_train'] = train_labels_for_epoch[:num_samples]
                predictions['predictions_val'] = val_predictions[:num_samples]
                predictions['labels_val'] = val_labels[:num_samples]
                predictions['predictions_test'] = test_predictions[:num_samples]
                predictions['labels_test'] = test_labels[:num_samples]

                dump_pickle(self.save_predictions, predictions)

                self.best_loss = val_loss
                overfit_counter = 0
            else:
                overfit_counter += 1
                if overfit_counter >= self.overfit_window:
                    print('stopping training since the model val_loss has been higher than the best loss: {} '
                          'for {} epochs'.format(self.best_loss, self.overfit_window))
                    break
