import argparse, os
import numpy as np

import tensorflow as tf
from keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import multi_gpu_model, to_categorical

import subprocess
import sys


# Script mode doesn't support requirements.txt

def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])


if __name__ == '__main__':

    # Keras-metrics brings additional metrics: precision, recall, f1
    install('keras-metrics')
    import keras_metrics

    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--dense-layer', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.2)

    parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])

    args, _ = parser.parse_known_args()

    epochs = args.epochs
    lr = args.learning_rate
    batch_size = args.batch_size
    dense_layer = args.dense_layer
    dropout = args.dropout

    gpu_count = args.gpu_count
    model_dir = args.model_dir
    training_dir = args.training
    validation_dir = args.validation

    x_train = np.load(os.path.join(training_dir, 'training.npz'))['image']
    y_train = np.load(os.path.join(training_dir, 'training.npz'))['label']
    x_val = np.load(os.path.join(validation_dir, 'test.npz'))['image']
    y_val = np.load(os.path.join(validation_dir, 'test.npz'))['label']

    # input image dimensions
    img_rows, img_cols = 28, 28

    # Tensorflow needs image channels last, e.g. (batch size, width, height, channels)
    if K.image_data_format() == 'channels_last':
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
        batch_norm_axis = -1
    else:

        print('Channels first, exiting')
        exit(-1)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_val.shape[0], 'test samples')

    # Normalize pixel values
    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_train /= 255
    x_val /= 255

    # Convert class vectors to binary class matrices
    num_classes = 10
    y_train = to_categorical(y_train, num_classes)
    y_val = to_categorical(y_val, num_classes)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        tf.keras.layers.Dense(32, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    if gpu_count > 1:
        model = multi_gpu_model(model, gpus=gpu_count)

    model.compile(loss=categorical_crossentropy,
                  optimizer=SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True),
                  metrics=['accuracy',
                           keras_metrics.precision(),
                           keras_metrics.recall(),
                           keras_metrics.f1_score()])

    # Only needed because of the BatchNormalization layer
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.local_variables_initializer())

    model.fit(x_train, y_train, batch_size=batch_size,
              validation_split=0.1,
              epochs=epochs)
    print(model.summary())
    score = model.evaluate(x_val, y_val, verbose=0)
    print('Validation loss    :', score[0])
    print('Validation accuracy:', score[1])

    # save Keras model for Tensorflow Serving
    model.save(os.path.join(model_dir, '1'))