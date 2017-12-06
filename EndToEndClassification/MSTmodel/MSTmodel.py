import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.python.ops import init_ops


class MSTmodel():
    """
    Neural network model that learns the logscaled Mel-spectrogram transformation.
    """

    def __init__(self, model_name, weights_initializer=initializers.xavier_initializer(),
                 biases_initializer=init_ops.zeros_initializer(), weights_regularizer=None, biases_regularizer=None):
        """
        Initializes the MSTmodel model class.
        
        Args:
            model_name (str): model name.
            num_classes (int): number of the classes (i.e. size of the output layer of the classifier).
            weights_initializer (func): how to initialize the weights of all layers.
            biases_initializer (func): how to initialize the biases of all layers.
            weights_regularizer (func): regularization of the weights of all layers.
            biases_regularizer (func): regularization of the biases of all layers.
        """

        self.model_name = model_name
        self.W_init = weights_initializer
        self.b_init = biases_initializer
        self.W_reg = weights_regularizer
        self.b_reg = biases_regularizer

    def build_predict_op(self, input_tensor, is_training=False):
        """
        Builds the graph from input tensor to model prediction. The 'is_training' argument is not used for now, but
        it allows easy handling of potential dropout/batchnorm layers.
                
        Args:
            input_tensor (tf tensor): input, with dimensions [batch_size, time, nr_channels=1].
            is_training (bool): whether in training mode (True) or evaluation mode (False)
        Returns:
            (tf operation): computes model predictions with dimensions [batch_size, mel_bands, time, nr_channels=1].
        """

        predict_op = input_tensor

        with tf.variable_scope('MSTmodel'):
            predict_op = slim.convolution(predict_op, 512, [1024], stride=[512], padding='SAME',
                                          activation_fn=None,
                                          weights_initializer=self.W_init, biases_initializer=self.b_init,
                                          weights_regularizer=self.W_reg, biases_regularizer=self.b_reg,
                                          scope='cnn_raw_1')
            predict_op = slim.batch_norm(predict_op, updates_collections=None, scope='cnn_raw_1',
                                            is_training=is_training)
            predict_op = tf.nn.relu(predict_op)
            predict_op = slim.convolution(predict_op, 256, [3], stride=[1], padding='SAME',
                                          activation_fn=None,
                                          weights_initializer=self.W_init, biases_initializer=self.b_init,
                                          weights_regularizer=self.W_reg, biases_regularizer=self.b_reg,
                                          scope='cnn_raw_2')
            predict_op = slim.batch_norm(predict_op, updates_collections=None, scope='cnn_raw_2',
                                            is_training=is_training)
            predict_op = tf.nn.relu(predict_op)
            predict_op = slim.convolution(predict_op, 60, [3], stride=[1], padding='SAME',
                                          activation_fn=None,
                                          weights_initializer=self.W_init, biases_initializer=self.b_init,
                                          weights_regularizer=self.W_reg, biases_regularizer=self.b_reg,
                                          scope='cnn_raw_3')
            predict_op = slim.batch_norm(predict_op, updates_collections=None, scope='cnn_raw_3',
                                            is_training=is_training)
            predict_op = tf.nn.tanh(predict_op)

        # transpose and add a channel dimension to match with the shape of the label
        predict_op = tf.transpose(predict_op, [0, 2, 1])
        predict_op = tf.expand_dims(predict_op, 3)

        return predict_op

    def get_loss_op(self, predictions, labels):
        """
        Builds the MSE loss op.
        
        Args:
            predictions (tf tensor): the batch of predicted log Mel-spectrograms.
            labels (tf tensor): the batch of label log Mel-spectrograms.
        
        Returns:
            (tf operation): computes the MSE loss operation averaged over the mini-batch.
        """

        loss = tf.losses.mean_squared_error(labels, predictions)

        return loss

    def save(self, path, sess):
        """
        Saves the model variables to the specified path.

        Args:
            path (str): folder path where the checkpoint will be saved.
            sess (tf Session): the session.
        """

        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="model")
        saver = tf.train.Saver(vars)
        saver.save(sess, path)

    def load(self, path, sess):
        """
        Loads the model variables from the specified path.

        Args:
            path (str): folder path from where the checkpoint will be loaded.
            sess (tf Session): the session.
        """

        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="model")
        saver = tf.train.Saver(vars)
        saver.restore(sess, path)
