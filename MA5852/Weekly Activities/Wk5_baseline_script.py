import argparse, os
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import multi_gpu_model

import subprocess
import sys


def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--dense-layer', type=int, default=512)

    parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])

    args, _ = parser.parse_known_args()

    epochs = args.epochs
    dense_layer = args.dense_layer

    gpu_count = args.gpu_count
    model_dir = args.model_dir
    training_dir = args.training
    validation_dir = args.validation

    X_train = np.load(os.path.join(training_dir, 'Wk5Training.npz'))['image']
    y_train = np.load(os.path.join(training_dir, 'Wk5Training.npz'))['label']
    X_val = np.load(os.path.join(validation_dir, 'Wk5Test.npz'))['image']
    y_val = np.load(os.path.join(validation_dir, 'Wk5Test.npz'))['label']

    print('X_train shape:', X_train.shape)
    print('y_train shape:', y_train.shape)
    print('X_valid shape:', X_val.shape)
    print('y_valid shape:', y_val.shape)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(500, input_dim=2, activation=tf.nn.relu),
        tf.keras.layers.Dense(3, activation=tf.nn.softmax)
    ])

    if gpu_count > 1:
        model = multi_gpu_model(model, gpus=gpu_count)

    model.compile(loss=binary_crossentropy,
                  optimizer=Adam(),
                  metrics=['accuracy'])

    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=epochs,
              verbose=1)

    print(model.summary())

    score = model.evaluate(X_val, y_val, verbose=0)
    print('Validation loss    :', score[0])
    print('Validation accuracy:', score[1])

    # save Keras model for Tensorflow Serving
    model.save(os.path.join(model_dir, '1'))
