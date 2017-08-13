import tensorflow as tf
import numpy as np
import os.path
import math

from Game import Game
from random import randint

game = Game()


input_layer = tf.placeholder(tf.float32, shape=(None, 9))
hidden = tf.layers.dense(inputs=input_layer, units=18, activation=tf.nn.tanh)
prediction = tf.layers.dense(inputs=hidden, units=9, activation=tf.nn.tanh)
expected = tf.placeholder(tf.float32, shape=(None, 9), name='expected')

train_loss = tf.reduce_mean(tf.losses.mean_squared_error(expected, prediction))
train_step = tf.train.AdagradOptimizer(0.2).minimize(train_loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

saver = tf.train.Saver()
if os.path.exists('train/train.ckpt.meta'):
    print('loading from train/train.ckpt')
    saver.restore(sess, "train/train.ckpt")


X = np.zeros((1, 9))
Y = np.zeros((1, 9))

illegal_moves_made = 0
won = 0
lost = 0
games = 1

illegal_moves_made = 0
won = 0
lost = 0
games = 1

game = Game()

for i in range(10000):
    ss = game.get_state()
    a = sess.run(prediction, feed_dict={input_layer: ss})
    aa = np.argmax(a)

    # mask = np.where(ss != 0, -2, 0)
    # a = a + mask
    aa = np.argmax(a)

    game.move(int(aa / 3), int(aa % 3), 1)
    rr = game.get_score()

    # play other player
    if math.fabs(rr) != 1:
        foo = game.get_state()
        foo = np.where(foo == 0)[1]
        if len(foo) > 0:
            foo = np.random.choice(foo, 1)[0]
            game.move(int(foo / 3), int(foo % 3), -1)

    foo = game.get_state()
    foo = np.where(foo == 0)[1]

    if math.fabs(rr) >= 1 or len(foo) == 0:
        games = games + 1
        if game.illegal_move:
            illegal_moves_made = illegal_moves_made + 1
        if game.get_score() == 1:
            won = won + 1
        if game.get_score() == -1:
            lost = lost + 1
        print(won / games * 100, lost / games * 100, illegal_moves_made / games * 100)
        game = Game()


        # Initialize Q-values (Q(s,a)Q(s,a)) arbitrarily for all state-action pairs.
# For life or until learning is stopped...
    # Choose an action (aa) in the current world state (ss) based on current Q-value estimates (Q(s,⋅)Q(s,⋅)).
    # Take the action (aa) and observe the the outcome state (s′s′) and reward (rr).
    # Update Q(s,a):=Q(s,a)+α[r+γmaxa′Q(s′,a′)−Q(s,a)]Q(s,a):=Q(s,a)+α[r+γmaxa′Q(s′,a′)−Q(s,a)]

