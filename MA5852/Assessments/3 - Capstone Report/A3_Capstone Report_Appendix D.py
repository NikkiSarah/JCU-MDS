# import the necessary libraries
import argparse
import json
import logging
import math
import numpy as np
import os
import pandas as pd
import subprocess
import sys

import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras import metrics
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import glorot_uniform, he_uniform
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import multi_gpu_model


def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])


if __name__ == '__main__':

    install('keras-metrics')
    import keras_metrics
    
    logging.getLogger().setLevel(logging.INFO)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    
    #tf.compat.v1.disable_eager_execution()
    
    parser = argparse.ArgumentParser()

    # add arguments for environment variables and any parameters/hyperparameters to be tuned
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--early-stop', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--dense-layer', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--model-name', type=str, default="baseline model")
    parser.add_argument('--optimiser', type=str, default=RMSprop)

    parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output-dir', type=str, default=os.environ['SM_OUTPUT_DIR'])
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--tf-logs-path', type=str, default=None)
    parser.add_argument('--training-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--validation-dir', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])

    args, _ = parser.parse_known_args()

    BATCH_SIZE = args.batch_size
    EARLYSTOP = args.early_stop
    EPOCHS = args.epochs
    DENSE_LAYER = args.dense_layer
    LEARNING_RATE = args.learning_rate
    OPTIMISER = args.optimiser
    MOMENTUM = args.momentum
    MODEL_NAME = args.model_name

    GPU_COUNT = args.gpu_count
    MODEL_DIR = args.model_dir
    OUTPUT_DIR = args.output_dir
    OUTPUT_DATA_DIR = args.output_data_dir
    TRAINING_DIR = args.training_dir
    VALIDATION_DIR = args.validation_dir

    # initialising TensorFlow summary writer
    job_name = json.loads(os.environ.get("SM_TRAINING_ENV"))["job_name"]
    logs_dir = "{}/{}".format(args.tf_logs_path, job_name)
    logging.info("Writing TensorBoard logs to {}".format(logs_dir))
    tf_writer = tf.summary.create_file_writer(logs_dir)
    tf_writer.set_as_default()
    
    # create a data generator and prepare training and validation iterators
    datagen = ImageDataGenerator(rescale=1.0/255.0)
    img_cols = 227
    img_rows = 227
    
    train_batches = datagen.flow_from_directory(TRAINING_DIR, class_mode='binary', batch_size=BATCH_SIZE,
                                                target_size=(img_rows, img_cols))
    val_batches = datagen.flow_from_directory(VALIDATION_DIR, class_mode='binary', batch_size=BATCH_SIZE,
                                              target_size=(img_rows, img_cols))

    # check the shape of the iterators as Tensorflow requires the channels to be last
    X_train = train_batches[0][0]
    y_train = train_batches[0][1]
    X_val = val_batches[0][0]
    y_val = val_batches[0][1]

    if K.image_data_format() == 'channels_last':
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
        input_shape = (img_rows, img_cols, 3)
        batch_norm_axis = -1
    else:
        print('Channels first, exiting')
        exit(-1)

    print('X_train shape:', X_train.shape)
    print('X_val shape:', X_val.shape)

    print('y_train shape:', y_train.shape)
    print('y_val shape:', y_val.shape)

    # add model assessment metrics
    METRICS = [
        metrics.TrueNegatives(name="tn"),
        metrics.FalseNegatives(name="fn"),
        metrics.TruePositives(name="tp"),
        metrics.FalsePositives(name="fp"),
        metrics.BinaryAccuracy(name="accuracy"),
        metrics.Precision(name="precision"),
        metrics.Recall(name="recall"),
        metrics.AUC(name="auc"),
        metrics.BinaryCrossentropy(name="loss")
    ]

    # define the CNN model
    def keras_model_fn(learning_rate=LEARNING_RATE,
                       optimiser=OPTIMISER,
                       momentum=MOMENTUM,
                       dense_layer=DENSE_LAYER,
                       metrics=METRICS,
                       model_name=MODEL_NAME):
        model = Sequential(name=model_name)
        model.add(Input(shape=(227, 227, 3)))
        model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same",
                         activation="relu", kernel_initializer="he_uniform"))
        model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same", 
                         activation="relu", kernel_initializer="he_uniform"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same",
                         activation="relu", kernel_initializer="he_uniform"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(dense_layer, activation="relu", kernel_initializer="he_uniform"))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        if GPU_COUNT > 1:
            model = multi_gpu_model(model, gpus=GPU_COUNT)

        # add the different optimiser options for model compilation
        if optimiser == 'SGD':
            model.compile(optimizer=SGD(learning_rate=learning_rate, momentum=momentum, nesterov=True),
                          loss='binary_crossentropy', metrics=metrics)
        elif optimiser == 'RMSprop':
            model.compile(optimizer=RMSprop(learning_rate=learning_rate, momentum=momentum),
                          loss='binary_crossentropy', metrics=metrics)
        elif optimiser == 'Nadam':
            model.compile(optimizer=Nadam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=metrics)
        else:
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=metrics)

        return model

    # run the model function
    model = keras_model_fn()
    # print a model summary
    print(model.summary())

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.local_variables_initializer())

    # setting the default parameters for early-stopping
    EARLYSTOP = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    tensorboard_cb = TensorBoard(log_dir=logs_dir, update_freq="epoch", histogram_freq=1, profile_batch="5,35")

    # fit the model to the training data and validate on the validation data
    # note that 28,000 and 6,000 are specific to the dataset being used, they are the sample size of the training and validation
    # datsets and should be changed accordingly
    num_training_batches = math.ceil(28000/BATCH_SIZE)
    num_validation_batches = math.ceil(6000/BATCH_SIZE)
    
    model.fit_generator(train_batches, validation_data=val_batches, epochs=EPOCHS, steps_per_epoch=len(train_batches),
                        validation_steps=len(val_batches), verbose=1, callbacks=[EARLYSTOP])

    train_scores = model.evaluate(train_batches, verbose=1)
    validation_scores = model.evaluate(val_batches, verbose=1)

    # print the final model metrics once the training job completes
    print('Training loss       :', train_scores[0])
    print('Training accuracy   :', train_scores[5])
    print('Training precision  :', train_scores[6])
    print('Training recall     :', train_scores[7])
    print('Training auc        :', train_scores[8])
    print('Validation loss     :', validation_scores[0])
    print('Validation accuracy :', validation_scores[5])
    print('Validation precision:', validation_scores[6])
    print('Validation recall   :', validation_scores[7])
    print('Validation auc      :', validation_scores[8])
    
    print('Training tn  :', train_scores[1])
    print('Training fn  :', train_scores[2])
    print('Training tp  :', train_scores[3])
    print('Training fp  :', train_scores[4])
    print('Validation tn:', validation_scores[1])
    print('Validation fn:', validation_scores[2])
    print('Validation tp:', validation_scores[3])
    print('Validation fp:', validation_scores[4])

    # save Keras model for Tensorflow Serving
    model.save(os.path.join(MODEL_DIR, '1'))