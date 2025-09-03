import numpy as np
import tensorflow.compat.v1 as tf

def forward_prop(prev, layers, activations, epsilon):
    #all layers get batch_normalization but the last one, that stays without any activation or normalization
    layer_prev = prev
    init =  tf.keras.initializers.VarianceScaling(mode='fan_avg')

    for i in range(len(layers) - 1):
        layer = tf.keras.layers.Dense(
                units = layers[i],
                kernel_initializer = init)

        mean, variance = tf.nn.moments(layer(layer_prev), axes = [0])

        gamma = tf.Variable(tf.ones(shape = (1, layers[i])), trainable = True)
        beta = tf.Variable(tf.zeros(shape = (1, layers[i])), trainable = True)

        batch_norm = tf.nn.batch_normalization(
            x = layer(layer_prev), 
            mean = mean, 
            variance = variance, 
            offset = beta, 
            scale = gamma, 
            variance_epsilon = epsilon)

        layer_prev = activations[i](batch_norm)

    output_layer = tf.keras.layers.Dense(units = layers[-1], kernel_initializer = init, activation = None)
    return output_layer(layer_prev)


def shuffle_data(X, Y):
    # fill the function
    indecies = np.random.permutation(X.shape[0])
    return X[indecies], Y[indecies]

def create_placeholders(nx, classes):
    x = tf.placeholder(dtype="float32", shape=(None, nx), name="x")
    y = tf.placeholder(dtype="float32", shape=(None, classes), name="y")
    return x, y

def calculate_accuracy(y, y_pred):
    predictions = tf.math.equal(tf.argmax(y, axis = 1), tf.argmax(y_pred, axis = 1))
    return tf.reduce_mean(tf.cast(predictions, tf.float32))

def calculate_loss(y, y_pred):
    return tf.losses.softmax_cross_entropy(onehot_labels = y, logits = y_pred)

def create_train_op(loss, alpha, beta1, beta2, epsilon):
    optimizer = tf.train.AdamOptimizer(
        learning_rate = alpha,
        beta1 = beta1,
        beta2 = beta2,
        epsilon = epsilon)
    train_op = optimizer.minimize(loss)
    return train_op

def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
        return tf.train.inverse_time_decay(
            learning_rate = alpha,
            global_step = global_step,
            decay_steps = decay_step,
            decay_rate = decay_rate,
            staircase = True)

def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    # get X_train, Y_train, X_valid, and Y_valid from Data_train and Data_valid
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid

    # initialize x, y and add them to collection
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    tf.add_to_collection(name = 'x', value = x)
    tf.add_to_collection(name = 'y', value = y)

    # initialize y_pred and add it to collection
    y_pred = forward_prop(x, layers, activations, epsilon)
    tf.add_to_collection(name = 'y_pred', value = y_pred)

    # intialize loss and add it to collection
    loss = calculate_loss(y, y_pred)
    tf.add_to_collection(name = 'loss', value = loss)

    # intialize accuracy and add it to collection
    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection(name = 'accuracy', value = accuracy)

    # intialize global_step variable
    # hint: not trainable
    global_step = tf.Variable(0, trainable = False)

    # compute decay_steps
    decay_steps = 1

    # create "alpha" the learning rate decay operation in tensorflow
    alpha = learning_rate_decay(alpha, decay_rate, global_step, decay_steps)

    # initizalize train_op and add it to collection
    # hint: don't forget to add global_step parameter in optimizer().minimize()
    # initizalize train_op and add it to collection
    train_op = create_train_op(loss, alpha, beta1, beta2, epsilon)
    tf.add_to_collection('train_op', train_op)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        train_dict = {
            x : X_train,
            y : Y_train
        }

        validation_dict = {
            x : X_valid,
            y : Y_valid
        }

        for i in range(epochs + 1):
            # print training and validation cost and accuracy
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

            # shuffle data
            X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)

            step = 0
            for j in range(0, X_train.shape[0], batch_size):
                # get X_batch and Y_batch from X_train shuffled and Y_train shuffled
                batch_len = batch_size
                if j + batch_size >= X_train.shape[0]:
                    batch_len = j + batch_size - X_train.shape[0]
                X_batch = X_shuffled[j : j + batch_len]
                Y_batch = Y_shuffled[j : j + batch_len]

                # run training operation
                mini_dict = {
                    x : X_batch,
                    y : Y_batch
                }
                sess.run(train_op, feed_dict = mini_dict)

                                # print batch cost and accuracy
                step += 1
                if step % 100 == 0:
                    step_cost, step_accuracy = sess.run([loss, accuracy] , feed_dict = mini_dict)
                    print("\tStep {0}".format(step))
                    print("\t\tCost: {0}".format(step_cost))
                    print("\t\tAccuracy: {0}".format(step_accuracy))
            sess.run(tf.assign(global_step, global_step + 1))


        # save and return the path to where the model was saved
        return tf.train.Saver().save(sess, save_path)

