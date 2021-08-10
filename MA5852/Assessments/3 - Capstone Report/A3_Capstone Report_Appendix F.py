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
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Nadam
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

    # add arguments for environment variables
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--model-name', type=str, default="inceptionV3 model")
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output-dir', type=str, default=os.environ['SM_OUTPUT_DIR'])
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--tf-logs-path', type=str, default=None)
    parser.add_argument('--training-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--validation-dir', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    
    args, _ = parser.parse_known_args()

    BATCH_SIZE = args.batch_size
    MODEL_NAME = args.model_name
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
    datagen = ImageDataGenerator(rescale=(1.0/127.5)-1)
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

    # define the inception model
    def keras_model_fn(model_name=MODEL_NAME,
                       metrics=METRICS):

        # add the InceptionV3 base
        inception_model = InceptionV3(input_shape = (227, 227, 3),
                                      include_top = False, # Leave out the last fully connected layer
                                      weights = 'imagenet')
        # set all layers to be non-trainable (note that this could increase the possibility over overfitting)
        for layer in inception_model.layers:
            layer.trainable = False

        # flatten the output layer to one dimension
        flatten = Flatten()(inception_model.output)
        # add a fully connected layer with 256 hidden units, ReLU activation and he_uniform initialisation
        dense = Dense(256, activation='relu', kernel_initializer='he_uniform')(flatten)
        # add a dropout rate of 0.5
        dropout = Dropout(0.5)(dense)
        # add a final sigmoid layer for classification
        classification = Dense(1, activation='sigmoid')(dropout)           

        # define the model
        model = Model(inputs=inception_model.input, outputs=classification)

        model.compile(optimizer=Nadam(learning_rate=0.0001), loss='binary_crossentropy', metrics=metrics)

        return model
    
    # run the model function
    model = keras_model_fn()
    # print a model summary
    print(model.summary())

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.local_variables_initializer())

    tensorboard_cb = TensorBoard(log_dir=logs_dir, update_freq="epoch", histogram_freq=1, profile_batch="5,35")

    # fit the model to the training data and validate on the validation data
    # note that 28,000 and 6,000 are specific to the dataset being used, they are the sample size of the training and validation
    # datsets and should be changed accordingly.
    num_training_batches = math.ceil(28000/BATCH_SIZE)
    num_validation_batches = math.ceil(6000/BATCH_SIZE)
    
    model.fit_generator(train_batches, validation_data=val_batches, epochs=20, steps_per_epoch=num_training_batches,
                        validation_steps=num_validation_batches, verbose=1)

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
    model.save(os.path.join(MODEL_DIR, '2'))