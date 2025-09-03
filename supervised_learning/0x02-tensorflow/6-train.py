#!/usr/bin/env python3

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop

def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    tf.add_to_collection(name = 'x', value = x)
    tf.add_to_collection(name = 'y', value = y)

    y_pred = forward_prop(x, layer_sizes, activations)
    tf.add_to_collection(name = 'y_pred', value = y_pred)

    loss = calculate_loss(y, y_pred)
    tf.add_to_collection(name = 'loss', value = loss)

    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection(name = 'accuracy', value = accuracy)

    train_op = create_train_op(loss, alpha)
    tf.add_to_collection(name = 'train_op', value = train_op)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    train_dict = {
        x : X_train,
        y : Y_train
    }

    validation_dict = {
        x : X_valid,
        y : Y_valid
    }

    sess = tf.Session()
    sess.run(init)
    step = 100
    i = 0
    while i <= iterations:
        train_cost = sess.run(loss, feed_dict = train_dict)
        train_acc = sess.run(accuracy, feed_dict = train_dict)
        val_cost = sess.run(loss, feed_dict = validation_dict)
        val_acc = sess.run(accuracy, feed_dict = validation_dict)

        if i % step == 0 or i == iterations:
            print("After {} iterations:".format(i))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_acc))
            print("\tValidation Cost: {}".format(val_cost))
            print("\tValidation Accuracy: {}".format(val_acc))
        if i < iterations:
            sess.run(train_op, feed_dict = train_dict)
        print(i)
        i += 1
    save_path = saver.save(sess, save_path)
    sess.close()
    return save_path
