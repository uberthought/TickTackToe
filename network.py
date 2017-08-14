import tensorflow as tf
import numpy as np
import os.path
import math

from Game import Game
from random import randint


class DNN:
    def __init__(self, state_size, action_size):
        self.input_layer = tf.placeholder(tf.float32, shape=(None, state_size))
        self.hidden = tf.layers.dense(inputs=self.input_layer, units=state_size, activation=tf.nn.tanh)
        self.keep_prob = tf.placeholder_with_default(1.0, [])
        self.dropout = tf.nn.dropout(self.hidden, self.keep_prob)
        self.prediction = tf.layers.dense(inputs=self.dropout, units=action_size, activation=tf.nn.tanh)

        self.expected = tf.placeholder(tf.float32, shape=(None, action_size))

        self.train_loss = tf.reduce_mean(tf.losses.mean_squared_error(self.expected, self.prediction))
        self.train_step = tf.train.GradientDescentOptimizer(0.2).minimize(self.train_loss)

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.saver = tf.train.Saver()
        if os.path.exists('train/train.ckpt.meta'):
            print('loading from train/train.ckpt')
            self.saver.restore(self.sess, "train/train.ckpt")

    def train(self, X, Y):
            loss = 1
            while loss >= 0.05:
                loss, _ = self.sess.run([self.train_loss, self.train_step], feed_dict={self.input_layer: X, self.expected: Y, self.keep_prob: 0.9})

    def run(self, X):
        return self.sess.run(self.prediction, feed_dict={self.input_layer: X})

    def save(self):
        self.saver.save(self.sess, "train/train.ckpt")

