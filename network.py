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
        self.keep_prob = tf.placeholder_with_default(1.0, [])
        self.stddev = tf.placeholder_with_default(0.0, [])

        self.input_layer = tf.placeholder(tf.float32, shape=(None, state_size))

        noise_generator = tf.random_normal(shape=tf.shape(self.input_layer), mean=0.0, stddev=self.stddev, dtype=tf.float32)
        self.noise1 = tf.add(self.input_layer, noise_generator)

        self.hidden1 = tf.layers.dense(inputs=self.noise1, units=state_size, activation=lrelu)
        self.dropout1 = tf.nn.dropout(self.hidden1, self.keep_prob)

        self.hidden2 = tf.layers.dense(inputs=self.hidden1, units=state_size * 2, activation=lrelu)
        self.dropout2 = tf.nn.dropout(self.hidden2, self.keep_prob)

        self.prediction = tf.layers.dense(inputs=self.dropout2, units=action_size)

        self.expected = tf.placeholder(tf.float32, shape=(None, action_size))

        self.train_loss = tf.reduce_mean(tf.losses.mean_squared_error(self.expected, self.prediction))
        # self.train_step = tf.train.GradientDescentOptimizer(0.01).minimize(self.train_loss)
        self.train_step = tf.train.AdagradOptimizer(0.1).minimize(self.train_loss)

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.saver = tf.train.Saver()
        if os.path.exists('train/train.ckpt.meta'):
            print('loading from train/train.ckpt')
            self.saver.restore(self.sess, "train/train.ckpt")

    def train(self, X, Y):
        feed_dict = {self.input_layer: X, self.expected: Y, self.keep_prob: 0.9, self.stddev: 0.1}
        # loss = self.sess.run(self.train_loss, feed_dict=feed_dict)
        loss = 100
        i = 0
        while i < 1000 and loss > 25:
            i += 1
            loss, _ = self.sess.run([self.train_loss, self.train_step], feed_dict=feed_dict)
            # if i != 0 and i % 100 == 0:
            #     print(loss)

    def run(self, X):
        return self.sess.run(self.prediction, feed_dict={self.input_layer: X})

    def save(self):
        self.saver.save(self.sess, "train/train.ckpt")
