import tensorflow as tf
import numpy as np
import os.path
import math

from Game import Game
from random import randint


def lrelu(x):
    alpha = 0.01
    return tf.maximum(alpha * x, x)


class DNN:
    def __init__(self, state_size, action_size):

        self.input_layer = tf.placeholder(tf.float32, shape=(None, state_size))
        self.hidden1 = tf.layers.dense(inputs=self.input_layer, units=state_size, activation=tf.nn.relu)
        # self.hidden2 = tf.layers.dense(inputs=self.input_layer, units=state_size, activation=tf.nn.tanh)
        self.prediction = tf.layers.dense(inputs=self.hidden1, units=action_size)

        self.expected = tf.placeholder(tf.float32, shape=(None, action_size))

        self.train_loss = tf.reduce_mean(tf.losses.mean_squared_error(self.expected, self.prediction))
        self.train_step = tf.train.AdagradOptimizer(0.2).minimize(self.train_loss)

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.saver = tf.train.Saver()
        self.path = 'train/train.ckpt'
        if os.path.exists(self.path + '.meta'):
            print('loading from ' + self.path)
            self.saver.restore(self.sess, self.path)

    def train(self, X, Y):
        feed_dict = {self.input_layer: X, self.expected: Y}
        # loss = self.sess.run(self.train_loss, feed_dict=feed_dict)
        loss = 1000
        i = 0
        while i < 1000 and loss > 50:
            i += 1
            loss, _ = self.sess.run([self.train_loss, self.train_step], feed_dict=feed_dict)
            if i != 0 and i % 100 == 0:
                print(loss)
        # print(loss)

    def run(self, X):
        return self.sess.run(self.prediction, feed_dict={self.input_layer: X})

    def save(self):
        self.saver.save(self.sess, self.path)
