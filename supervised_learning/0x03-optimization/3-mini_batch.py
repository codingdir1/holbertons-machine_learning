#!/usr/bin/env python3

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

shuffle_data = __import__('2-shuffle_data').shuffle_data

def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32, epochs=5, load_path="/tmp/model.ckpt", save_path="/tmp/model.ckpt"):
    
    loader = tf.train.import_meta_graph(load_path + ".meta")
    saver = tf.train.Saver()

    sess = tf.Session()
    loader.restore(sess, load_path)

    x = tf.get_collection("x")[0]
    y = tf.get_collection("y")[0]
    accuracy = tf.get_collection("accuracy")[0]
    loss = tf.get_collection("loss")[0]
    train_op = tf.get_collection("train_op")[0]

    train_dict = {
        x : X_train,
        y : Y_train
    }

    validation_dict = {
        x : X_valid,
        y : Y_valid
    }

    for i in range(epochs + 1):
        X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)

        train_cost = sess.run(loss, feed_dict = train_dict)
        train_accuracy = sess.run(accuracy, feed_dict = train_dict)
        valid_cost = sess.run(loss, feed_dict = validation_dict)
        valid_accuracy = sess.run(accuracy, feed_dict = validation_dict)

        print("After {0} epochs".format(i))
        print("\tTraining Cost: {0}".format(train_cost))
        print("\tTraining Accuracy: {0}".format(train_accuracy))
        print("\tValidation Cost: {0}".format(valid_cost))
        print("\tValidation Accuracy: {0}".format(valid_accuracy))

        if i == epochs:
            break

        step = 0
        offset = 0
        while offset < X_shuffled.shape[0]:
            batch_len = batch_size
            if X_shuffled.shape[0] - offset < batch_size:
                batch_len = X_shuffled.shape[0] - offset
            X_batch = X_shuffled[offset : offset + batch_len]
            Y_batch = Y_shuffled[offset : offset + batch_len]

            mini_dict = {
                x : X_batch,
                y : Y_batch
            }

            sess.run(train_op, feed_dict = mini_dict)
            step += 1

            if step % 100 == 0:
                step_cost, step_accuracy = sess.run([loss, accuracy] , feed_dict = mini_dict)

                print("\tStep {0}".format(step))
                print("\t\tCost: {0}".format(step_cost))
                print("\t\tAccuracy: {0}".format(step_accuracy))

            offset += batch_len

    save_path = saver.save(sess, save_path)
    sess.close()
    return save_path
