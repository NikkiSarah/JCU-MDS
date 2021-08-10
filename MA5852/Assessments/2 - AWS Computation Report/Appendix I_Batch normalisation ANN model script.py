import argparse, os
import logging
import numpy as np
import pandas as pd
import subprocess
import sys

import tensorflow as tf
from tensorflow.keras import (
    metrics,
    Model,
    optimizers)
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    PReLU)
from tensorflow.keras.utils import multi_gpu_model


def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])


if __name__ == '__main__':

    install('keras-metrics')
    import keras_metrics

    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=50)

    parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output-dir', type=str, default=os.environ['SM_OUTPUT_DIR'])
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--training-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--validation-dir', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])

    args, _ = parser.parse_known_args()

    EPOCHS = args.epochs

    GPU_COUNT = args.gpu_count
    MODEL_DIR = args.model_dir
    OUTPUT_DIR = args.output_dir
    OUTPUT_DATA_DIR = args.output_data_dir
    TRAINING_DIR = args.training_dir
    VALIDATION_DIR = args.validation_dir

    X_train = np.load(os.path.join(TRAINING_DIR, 'training.npz'))['features']
    y_train = np.load(os.path.join(TRAINING_DIR, 'training.npz'))['labels']
    X_val = np.load(os.path.join(VALIDATION_DIR, 'validation.npz'))['features']
    y_val = np.load(os.path.join(VALIDATION_DIR, 'validation.npz'))['labels']

    print('X_train shape:', X_train.shape)
    print('y_train shape:', y_train.shape)
    print('X_valid shape:', X_val.shape)
    print('y_valid shape:', y_val.shape)

    METRICS = [
        metrics.TrueNegatives(name="tn"),
        metrics.FalseNegatives(name="fn"),
        metrics.TruePositives(name="tp"),
        metrics.FalsePositives(name="fp"),
        metrics.BinaryAccuracy(name="accuracy"),
        metrics.Precision(name="precision"),
        metrics.Recall(name="recall"),
        metrics.AUC(name="auc"),
        metrics.BinaryCrossentropy(name="loss"),
        metrics.AUC(name='prc', curve='PR')
    ]


    def keras_model_fn(metrics=METRICS,
                       input_dim=X_train.shape[1]):
        input_layer = tf.keras.Input(shape=(input_dim,))
        x1 = BatchNormalization()(input_layer)
        x1 = Dense(40, use_bias=False)(x1)
        x1 = BatchNormalization()(x1)
        x1 = PReLU()(x1)
        x2 = Dense(44, use_bias=False)(x1)
        x2 = BatchNormalization()(x2)
        x2 = PReLU()(x2)
        x2 = BatchNormalization()(x2)
        output_layer = Dense(1, activation='sigmoid')(x2)

        model = Model(inputs=input_layer, outputs=output_layer, name='batch_normalised_model')

        if GPU_COUNT > 1:
            model = multi_gpu_model(model, gpus=GPU_COUNT)

        model.compile(optimizer=optimizers.Adam(0.01),
                      loss='binary_crossentropy',
                      metrics=metrics)

        return model


    model = keras_model_fn()
    print(model.summary())

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.local_variables_initializer())

    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=EPOCHS,
              batch_size=512,
              verbose=1)

    train_scores = model.evaluate(X_train, y_train, verbose=1)
    validation_scores = model.evaluate(X_val, y_val, verbose=1)

    print('Training loss       :', train_scores[0])
    print('Training accuracy   :', train_scores[5])
    print('Training precision  :', train_scores[6])
    print('Training recall     :', train_scores[7])
    print('Training auc        :', train_scores[8])
    print('Training prc        :', train_scores[10])
    print('Validation loss     :', validation_scores[0])
    print('Validation accuracy :', validation_scores[5])
    print('Validation precision:', validation_scores[6])
    print('Validation recall   :', validation_scores[7])
    print('Validation auc      :', validation_scores[8])
    print('Validation prc      :', validation_scores[10])

    # save Keras model for Tensorflow Serving
    model.save(os.path.join(MODEL_DIR, '1'))
