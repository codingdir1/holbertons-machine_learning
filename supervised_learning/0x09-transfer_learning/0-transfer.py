#!/usr/bin/env python3

import tensorflow.keras as K

def preprocess_data(X, Y):
    X_p = K.applications.densenet.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, num_classes = 10)

    return X_p, Y_p

def resize_images(X):
    return K.backend.resize_images(
            x = X,
            height_factor = 7,
            width_factor = 7,
            data_format = "channels_last")

(X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
xtrain, ytrain = preprocess_data(X_train, Y_train)
xtest, ytest = preprocess_data(X_test, Y_test)

base = K.applications.DenseNet121(
        include_top = False,
        weights = "imagenet",
        input_shape = (224, 224, 3))

for layer in base.layers[:100]:
    layer.trainable = False

X = K.Input(shape=(32, 32, 3))
Lambda = K.layers.Lambda(resize_images)(X)
base_1 = base(Lambda)
flatten = K.layers.Flatten()(base_1)
batch_normalization = K.layers.BatchNormalization(axis = -1)(flatten)
dense = K.layers.Dense(units = 10,
                       activation = "softmax")(batch_normalization)

calls = [K.callbacks.ModelCheckpoint(filepath='cifar10.h5',
                       save_best_only=True)]

model = K.Model(inputs = X, outputs = dense)

model.compile(optimizer='rmsprop',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

model.fit(xtrain, ytrain,
                        batch_size=32,
                        epochs=10,
                        validation_data=(xtest, ytest),
                        shuffle=True,
                        callbacks=calls)
