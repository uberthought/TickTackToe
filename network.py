import tensorflow as tf
import numpy as np
import os.path
import math

from Game import Game
from random import randint


def lrelu(x):
    alpha = 0.1
    return tf.maximum(alpha * x, x)


class DNN:
    def __init__(self, state_size, action_size):

        self.keep_prob = tf.placeholder_with_default(1.0, [])

        self.input_layer = tf.placeholder(tf.float32, shape=(None, state_size))

        self.hidden1 = tf.layers.dense(inputs=self.input_layer, units=state_size, activation=tf.nn.relu)
        self.dropout1 = tf.nn.dropout(self.hidden1, self.keep_prob)

        self.hidden2 = tf.layers.dense(inputs=self.dropout1, units=state_size, activation=tf.nn.relu)
        self.dropout2 = tf.nn.dropout(self.hidden2, self.keep_prob)

        self.prediction = tf.layers.dense(inputs=self.dropout2, units=action_size)

        self.expected = tf.placeholder(tf.float32, shape=(None, action_size))

        self.train_loss = tf.reduce_mean(tf.losses.mean_squared_error(self.expected, self.prediction))
        self.train_step = tf.train.AdagradOptimizer(.2).minimize(self.train_loss)

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.saver = tf.train.Saver()
        self.path = 'train/train.ckpt'
        if os.path.exists(self.path + '.meta'):
            print('loading from ' + self.path)
            self.saver.restore(self.sess, self.path)

    def train(self, X, Y):
        feed_dict = {self.input_layer: X, self.expected: Y, self.keep_prob: .9}
        # loss = self.sess.run(self.train_loss, feed_dict=feed_dict)
        loss = 1000
        i = 0
        while i < 20000 and loss > 100:
            i += 1
            loss, _ = self.sess.run([self.train_loss, self.train_step], feed_dict=feed_dict)
            if i != 0 and i % 2000 == 0:
                print('loss ', loss)

        return loss

    def run(self, X):
        return self.sess.run(self.prediction, feed_dict={self.input_layer: X})

    def save(self):
        self.saver.save(self.sess, self.path)
