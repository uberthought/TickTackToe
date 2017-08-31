import tensorflow as tf
import numpy as np
import os.path
import math
import logging

from Game import Game
from random import randint


def lrelu(x):
    alpha = 0.1
    return tf.maximum(alpha * x, x)


class DNN:

    keep_prob = tf.placeholder_with_default(1.0, [])

    input_layer = tf.placeholder(tf.float32, shape=(None, Game.state_size))

    hidden1 = tf.layers.dense(inputs=input_layer, units=Game.state_size, activation=tf.nn.tanh)
    dropout1 = tf.nn.dropout(hidden1, keep_prob)

    hidden2 = tf.layers.dense(inputs=dropout1, units=Game.state_size, activation=tf.nn.tanh)
    dropout2 = tf.nn.dropout(hidden2, keep_prob)

    prediction = tf.layers.dense(inputs=dropout2, units=Game.action_size)

    expected = tf.placeholder(tf.float32, shape=(None, Game.action_size))

    train_loss = tf.reduce_mean(tf.losses.mean_squared_error(expected, prediction))
    train_step = tf.train.AdagradOptimizer(.2).minimize(train_loss)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()

    path = 'train/train.ckpt'

    def __init__(self):

        if os.path.exists(DNN.path + '.meta'):
            logging.info('loading from ' + DNN.path)
            DNN.saver.restore(DNN.sess, DNN.path)

    def train(self, X, Y):
        feed_dict = {DNN.input_layer: X, DNN.expected: Y, DNN.keep_prob: .8}
        # loss = DNN.sess.run(train_loss, feed_dict=feed_dict)
        loss = 1000
        i = 0
        while i < 20000 and loss > 25:
        # while i < 100:
            i += 1
            loss, _ = DNN.sess.run([DNN.train_loss, DNN.train_step], feed_dict=feed_dict)
            if i != 0 and i % 2000 == 0:
                logging.debug('loss ', loss)

        return loss

    def run(self, X):
        return DNN.sess.run(DNN.prediction, feed_dict={DNN.input_layer: X})

    def save(self):
        DNN.saver.save(DNN.sess, DNN.path)
