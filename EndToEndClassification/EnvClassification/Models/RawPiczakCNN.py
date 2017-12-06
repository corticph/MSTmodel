import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.python.ops import init_ops


class RawPiczak():
    """
    PiczakCNN adapted to work with raw speech as input.
    
    For details, see the NIPS workshop paper. 
    """

    def __init__(self, model_name, num_classes=50, weights_initializer=initializers.xavier_initializer(),
                 biases_initializer=init_ops.zeros_initializer(), weights_regularizer=None, biases_regularizer=None,
                 dropout=False):
        """
        Initializes the RawPiczakCNN model class.
        
        Args:
            model_name (str): model name.
            num_classes (int): number of the classes (i.e. size of the output layer of the classifier).
            weights_initializer (func): how to initialize the weights of all layers.
            biases_initializer (func): how to initialize the biases of all layers.
            weights_regularizer (func): regularization of the weights of all layers.
            biases_regularizer (func): regularization of the biases of all layers.
            dropout (bool): whether or not to use dropout. For training with random initialization (so without loading
                            the pretrained MSTmodel layers dropout prevents overfitting.
        """

        self.model_name = model_name
        self.num_classes = num_classes
        self.W_init = weights_initializer
        self.b_init = biases_initializer
        self.W_reg = weights_regularizer
        self.b_reg = biases_regularizer
        self.dropout = dropout

    def build_predict_op(self, input_tensor, is_training=False):
        predict_op = input_tensor

        if not self.dropout:
            with tf.variable_scope('MSTmodel'):
                predict_op = slim.convolution(predict_op, 512, [1024], stride=[512], padding='SAME',
                                              activation_fn=None,
                                              weights_initializer=self.W_init, biases_initializer=self.b_init,
                                              weights_regularizer=self.W_reg, biases_regularizer=self.b_reg,
                                              scope='cnn_raw_1')
                predict_op = tf.nn.relu(predict_op)
                predict_op = slim.convolution(predict_op, 256, [3], stride=[1], padding='SAME',
                                              activation_fn=None,
                                              weights_initializer=self.W_init, biases_initializer=self.b_init,
                                              weights_regularizer=self.W_reg, biases_regularizer=self.b_reg,
                                              scope='cnn_raw_2')
                predict_op = tf.nn.relu(predict_op)
                predict_op = slim.convolution(predict_op, 60, [3], stride=[1], padding='SAME',
                                              activation_fn=None,
                                              weights_initializer=self.W_init, biases_initializer=self.b_init,
                                              weights_regularizer=self.W_reg, biases_regularizer=self.b_reg,
                                              scope='cnn_raw_3')
                predict_op = tf.nn.tanh(predict_op)

                # transpose and add a channel dimension to match with the shapes that the PiczakCNN expects
                predict_op = tf.transpose(predict_op, [0, 2, 1])
                predict_op = tf.expand_dims(predict_op, 3)
        else:
            # add dropout layers
            with tf.variable_scope('MSTmodel'):
                predict_op = slim.convolution(predict_op, 512, [1024], stride=[512], padding='SAME',
                                              activation_fn=None,
                                              weights_initializer=self.W_init, biases_initializer=self.b_init,
                                              weights_regularizer=self.W_reg, biases_regularizer=self.b_reg,
                                              scope='cnn_raw_1')
                predict_op = tf.nn.relu(predict_op)
                predict_op = slim.dropout(predict_op, keep_prob=0.5, is_training=is_training, scope='cnn_raw_1')
                predict_op = slim.convolution(predict_op, 256, [3], stride=[1], padding='SAME',
                                              activation_fn=None,
                                              weights_initializer=self.W_init, biases_initializer=self.b_init,
                                              weights_regularizer=self.W_reg, biases_regularizer=self.b_reg,
                                              scope='cnn_raw_2')
                predict_op = tf.nn.relu(predict_op)
                predict_op = slim.dropout(predict_op, keep_prob=0.5, is_training=is_training, scope='cnn_raw_2')
                predict_op = slim.convolution(predict_op, 60, [3], stride=[1], padding='SAME',
                                              activation_fn=None,
                                              weights_initializer=self.W_init, biases_initializer=self.b_init,
                                              weights_regularizer=self.W_reg, biases_regularizer=self.b_reg,
                                              scope='cnn_raw_3')
                predict_op = tf.nn.tanh(predict_op)
                predict_op = slim.dropout(predict_op, keep_prob=0.5, is_training=is_training, scope='cnn_raw_3')

                # transpose and add a channel dimension to match with what the PiczakCNN expects
                predict_op = tf.transpose(predict_op, [0, 2, 1])
                predict_op = tf.expand_dims(predict_op, 3)

        with tf.variable_scope('piczak'):
            # first convolutional block with dropout and with max pooling
            predict_op = slim.convolution(predict_op, 80, [57, 6], stride=[1, 1], padding='VALID', activation_fn=None,
                                          weights_initializer=self.W_init, biases_initializer=self.b_init,
                                          weights_regularizer=self.W_reg, biases_regularizer=self.b_reg, scope='cnn_1')
            predict_op = tf.nn.relu(predict_op)
            predict_op = slim.dropout(predict_op, keep_prob=0.5, is_training=is_training, scope='cnn_1')
            predict_op = slim.pool(predict_op, [4, 3], 'MAX', padding='VALID', stride=[1, 3])

            # second convolutional block without dropout (following Piczak) and with max pooling
            predict_op = slim.convolution(predict_op, 80, [1, 3], stride=[1, 1], padding='VALID', activation_fn=None,
                                          weights_initializer=self.W_init, biases_initializer=self.b_init,
                                          weights_regularizer=self.W_reg, biases_regularizer=self.b_reg, scope='cnn_2')
            predict_op = tf.nn.relu(predict_op)
            predict_op = slim.pool(predict_op, [1, 3], 'MAX', padding='VALID', stride=[1, 3])

            # reshaping before the dense layers
            predict_op = tf.transpose(predict_op, [0, 2, 1, 3])

            print('shape of output after reshaping')
            # print('should be: bs, 10, 1, num_filters=80')
            print(predict_op.get_shape())

            shx = predict_op.get_shape()
            predict_op = tf.reshape(predict_op, [-1, int(shx[1]), int(shx[2] * shx[3])])
            print('shape of output after reshaping')
            # print('should be: bs, 10, 80')
            print(predict_op.get_shape())

            shx = predict_op.get_shape()
            predict_op = tf.reshape(predict_op, [-1, int(shx[1]) * int(shx[2])])
            print('shape of output after another reshaping')
            # print('should be: bs, 800')
            print(predict_op.get_shape())

            # dense part of the model with dropout
            predict_op = slim.fully_connected(predict_op, 5000, activation_fn=None,
                                              weights_initializer=self.W_init, biases_initializer=self.b_init,
                                              weights_regularizer=self.W_reg, biases_regularizer=self.b_reg,
                                              scope='dense_1')
            predict_op = tf.nn.relu(predict_op)
            predict_op = slim.dropout(predict_op, keep_prob=0.5, is_training=is_training, scope='dense_1')

            predict_op = slim.fully_connected(predict_op, 5000, activation_fn=None,
                                              weights_initializer=self.W_init, biases_initializer=self.b_init,
                                              weights_regularizer=self.W_reg, biases_regularizer=self.b_reg,
                                              scope='dense_2')
            predict_op = tf.nn.relu(predict_op)
            predict_op = slim.dropout(predict_op, keep_prob=0.5, is_training=is_training, scope='dense_2')

            # linear output layer
            predict_op = slim.fully_connected(predict_op, self.num_classes, activation_fn=None,
                                              weights_initializer=self.W_init, biases_initializer=self.b_init,
                                              weights_regularizer=self.W_reg, biases_regularizer=self.b_reg,
                                              scope='output')

        return predict_op

    def get_loss_op(self, prediction, label_tensor):
        """
        Builds the cross entropy loss op.
        
        Args:
            prediction (tf tensor): model prediction with dimensions [batch_size, num_classes].
            label_tensor (tf tensor): integer labels (not one-hot encoded!) with dimension [batch_size] where
                                    each entry in labels must be an index in [0, num_classes).
        
        Returns:
            (tf operation): computes cross entropy loss op averaged over the mini-batch.
        """

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_tensor, logits=prediction)
        loss = tf.reduce_mean(loss)

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

    def save_MSTmodel(self, path, sess):
        """
        Saves the model variables to the specified path.

        Args:
            path (str): folder path where the checkpoint will be saved.
            sess (tf Session): the session.
        """

        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="model/MSTmodel")
        saver = tf.train.Saver(vars)
        saver.save(sess, path)

    def save_piczak(self, path, sess):
        """
        Saves the model variables to the specified path.

        Args:
            path (str): folder path where the checkpoint will be saved.
            sess (tf Session): the session.
        """

        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="model/piczak")
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

    def load_MSTmodel(self, path, sess):
        """
        Loads the model variables from the specified path.

        Args:
            path (str): folder path from where the checkpoint will be loaded.
            sess (tf Session): the session.
        """

        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="model/MSTmodel")
        saver = tf.train.Saver(vars)
        saver.restore(sess, path)

    def load_piczak(self, path, sess):
        """
        Loads the model variables from the specified path.

        Args:
            path (str): folder path from where the checkpoint will be loaded.
            sess (tf Session): the session.
        """

        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="model/piczak")
        saver = tf.train.Saver(vars)
        saver.restore(sess, path)
