import tensorflow as tf
import numpy as np
import os.path
import math

from Game import Game
from random import randint

game = Game()
# game.move(0, 0, -1)
# game.move(0, 2, 1)
# game.move(0, 1, -1)
# game.move(1, 1, 1)
# game.move(1, 0, -1)
# game.move(2, 2, 1)
# game.move(1, 2, -1)


input_layer = tf.placeholder(tf.float32, shape=(None, 9), name='input')
hidden = tf.layers.dense(inputs=input_layer, units=9, activation=tf.nn.tanh, name='hidden')
prediction = tf.layers.dense(inputs=hidden, units=9, activation=tf.nn.tanh, name='prediction')
expected = tf.placeholder(tf.float32, shape=(None, 9), name='expected')

train_loss = tf.reduce_mean(tf.losses.mean_squared_error(expected, prediction))
train_step = tf.train.AdagradOptimizer(0.5).minimize(train_loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

saver = tf.train.Saver()
if os.path.exists('train/train.ckpt.meta'):
    print('loading from train/train.ckpt')
    saver.restore(sess, "train/train.ckpt")


def train(X, Y):
    feed_dict={input_layer: X, expected: Y}

    loss = 1
    while loss > 0.1:
        loss, _= sess.run([train_loss, train_step], feed_dict=feed_dict)
        # print(loss)

    # feed_dict={input_layer: X, expected: Y}
    # prediction_output, expected_output= sess.run([prediction, expected], feed_dict=feed_dict)
    #
    # print(prediction_output)
    # print(expected_output)

# X = game.get_state()
# Y = np.abs(np.copy(X)) * -1
# Y[0][1] = 1

# for i in range(1000):
#     loss, _ = sess.run([train_loss, train_step], feed_dict={input_layer: X, expected: Y})
    #     print(loss)
#
# prediction_output, expected_output= sess.run([prediction, expected], feed_dict={input_layer: X, expected: Y})
# a = np.argmax(prediction_output)
#
# print(prediction_output)
# print(expected_output)
# print(a)
#
# print(game.state)
# game.move(int(a % 3), int(a / 3), 1)
# print(game.state)
#
# exit()

X = np.zeros((1, 9))
Y = np.zeros((1, 9))

illegal_moves_made = 0
won = 0
lost = 0
games = 1

# For life or until learning is stopped...
for i in range(10):

    # print(game.state)

    # Choose an action (aa) in the current world state (ss) based on current Q-value estimates (Q(s,⋅)Q(s,⋅)).
    ss = game.get_state()
    a = sess.run(prediction, feed_dict={input_layer: ss})

    if randint(0,9) == 0:
        a = np.random.rand(1,9) + -0.5
    mask = np.where(ss != 0, -2, 0)
    aa = np.argmax(a + mask)

    # Take the action (aa) and observe the the outcome state (s′s′) and reward (rr).
    game.move(int(aa / 3), int(aa % 3), 1)
    rr = game.get_score()

    # print()
    # print('State')
    # print(game.get_state())
    # print('a')
    # print(a)
    # print('rr ', rr)

    # store result
    # a[0][aa] = rr + 0.1 * a[0][aa]
    a[0][aa] = rr

    X = np.concatenate((X, ss), axis=0)
    Y = np.concatenate((Y, a), axis=0)
    if len(X) > 1000:
        X = X[-1000:, ]
        Y = Y[-1000:, ]

    # play other player
    if math.fabs(rr) != 1:
        foo = game.get_state()
        foo = np.where(foo == 0)[1]
        if len(foo) > 0:
            foo = np.random.choice(foo, 1)[0]
            # print('Random move: ', foo)
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

        train(X, Y)

        # X = np.zeros((1, 9))
        # Y = np.zeros((1, 9))
        game = Game()

    if i % 1000 == 0:
        saver.save(sess, "train/train.ckpt")

illegal_moves_made = 0
won = 0
lost = 0
games = 1

game = Game()

for i in range(10000):
    ss = game.get_state()
    a = sess.run(prediction, feed_dict={input_layer: ss})
    aa = np.argmax(a)

    mask = np.where(ss != 0, -2, 0)
    aa = np.argmax(a + mask)

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

