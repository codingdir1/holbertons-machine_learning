#!/usr/bin/env python3

import tensorflow.compat.v1 as tf

forward_prop = __import__('2-forward_prop').forward_prop

def evaluate(X, Y, save_path):

    saver = tf.train.import_meta_graph(save_path + ".meta")

    sess = tf.Session()
    saver.restore(sess, save_path)

    x = tf.get_collection("x")[0]
    y = tf.get_collection("y")[0]
    y_pred = tf.get_collection("y_pred")[0]
    accuracy = tf.get_collection("accuracy")[0]
    loss = tf.get_collection("loss")[0]

    eval_dict = {
            x: X,
            y: Y
    }

    y_eval = sess.run(y_pred, feed_dict=eval_dict)
    eval_acc = sess.run(accuracy, feed_dict=eval_dict)
    eval_loss = sess.run(loss, feed_dict=eval_dict)
    sess.close()
    return y_eval, eval_acc, eval_loss
