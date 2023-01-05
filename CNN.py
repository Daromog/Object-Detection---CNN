#Version .........................................

#data_base=228928 images
#train_Data=160249  CA=80144 SA=80105
#val_Data=34339    CA=17110 SA=17229 
#test_Data=34339   CA=17110 SA=17229


#Imports......................................................

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import argparse
from tensorflow.contrib.learn.python.learn.utils import (saved_model_export_utils)
from tensorflow.contrib.training.python.training import hparam



#Command to print info of training
tf.logging.set_verbosity(tf.logging.INFO)


#Convolutional Funcion .......................................

def cnn_model_fn(features, labels, mode):

    """Model"""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 224x224 pixels, and have 3 color channel
    input_layer = tf.reshape(features["image"], [-1, 224, 224, 3])


    #Command to see images in tensorboard
    tf.summary.image('input',input_layer,1)


    # Convolutional Layer #1
    # Computes 64 features using a 3x3 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 224, 224, 3]
    # Output Tensor Shape: [batch_size, 224, 224, 64]
    conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=64,
      kernel_size=[7, 7],
      padding="same",
      activation=tf.nn.relu)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 224, 224, 64]
    # Output Tensor Shape: [batch_size, 112, 112, 64]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    # Computes 128 features using a 3x3 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 112, 112, 64]
    # Output Tensor Shape: [batch_size, 112, 112, 128]
    conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=128,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 112, 112,128 ]
    # Output Tensor Shape: [batch_size, 56, 56, 128]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)


    # Convolutional Layer #3
    # Computes 256 features using a 3x3 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 56, 56, 128]
    # Output Tensor Shape: [batch_size, 56, 56, 256]
    conv3 = tf.layers.conv2d(
      inputs=pool2,
      filters=256,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

    # Pooling Layer #3
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 56, 56, 256]
    # Output Tensor Shape: [batch_size, 28, 28, 256]
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    # Convolutional Layer #4
    # Computes 512 features using a 3x3 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 28, 28, 256]
    # Output Tensor Shape: [batch_size, 28, 28, 512]
    conv4 = tf.layers.conv2d(
      inputs=pool3,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

    # Pooling Layer #4
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 28, 28, 512]
    # Output Tensor Shape: [batch_size, 14, 14, 512]
    pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

    # Convolutional Layer #5
    # Computes 512 features using a 3x3 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 14, 14, 512]
    # Output Tensor Shape: [batch_size, 14, 14, 512]
    conv5 = tf.layers.conv2d(
      inputs=pool4,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

    # Pooling Layer #5
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 14, 14, 512]
    # Output Tensor Shape: [batch_size, 7, 7, 512]
    pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 7, 7, 512]
    # Output Tensor Shape: [batch_size, 7 * 7 * 512]
    pool5_flat = tf.reshape(pool5, [-1, 7 * 7 * 512])


    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(inputs=pool5_flat, units=4096, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    #Final Layer with 2 outputs
    logits = tf.layers.dense(inputs=dropout, units=2)


    #Calculate the loss
    onehot_labels=tf.one_hot(indices=labels, depth=2)
    loss=tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    #Predictions
    output=tf.nn.softmax(logits)
    predictions=tf.argmax(input=output, axis=1)
    max_label=tf.argmax(input=onehot_labels, axis=1)

    #o_hot = tf.train.LoggingTensorHook(tensors={"one":onehot_labels}, every_n_iter=1)
    #m=tf.train.LoggingTensorHook(tensors={"max":max_label}, every_n_iter=1)
    #l=tf.train.LoggingTensorHook(tensors={"logits":tf.nn.softmax(logits)}, every_n_iter=1)
    #p=tf.train.LoggingTensorHook(tensors={"predictions":predictions},every_n_iter=1)

    #Calculate the accuracy
    
    accuracy=tf.metrics.accuracy(labels=max_label,predictions=predictions,name='acc_op')
    metrics={"accuracy":accuracy}
    tf.summary.scalar("accuracy",accuracy[1])


    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Configure
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode,loss=loss,eval_metric_ops=metrics)


#Function to read the validation record.........................................
def val_input_fn():
    filenames=["gs://pruebasfinales-222203-mlengineecho/valgrayrgb.tfrecord"]
    dataset=tf.data.TFRecordDataset(filenames)

    def parser(record):
        keys_to_features={
            "image_raw":tf.FixedLenFeature([],tf.string),
            "label":tf.FixedLenFeature([],tf.int64)
        }
        parsed=tf.parse_single_example(record,keys_to_features)

        image=tf.decode_raw(parsed["image_raw"],tf.uint8)
        image=tf.cast(image,tf.float32)
        image=tf.reshape(image,[224,224,3])
        label=tf.cast(parsed["label"],tf.int32)

        return {"image":image},label

    dataset=dataset.map(parser)
    dataset=dataset.shuffle(buffer_size=500)
    dataset=dataset.batch(200)
    dataset=dataset.repeat(1)


    return dataset

#Function to read the training record.........................................
def train_input_fn():
    filenames=["gs://pruebasfinales-222203-mlengineecho/trainrgb.tfrecord"]
    dataset=tf.data.TFRecordDataset(filenames)

    def parser(record):
        keys_to_features={
            "image_raw":tf.FixedLenFeature([],tf.string),
            "label":tf.FixedLenFeature([],tf.int64)
        }
        parsed=tf.parse_single_example(record,keys_to_features)

        image=tf.decode_raw(parsed["image_raw"],tf.uint8)
        image=tf.cast(image,tf.float32)
        image=tf.reshape(image,[224,224,3])
        label=tf.cast(parsed["label"],tf.int32)

        return {"image":image},label

    dataset=dataset.map(parser)
    dataset=dataset.shuffle(buffer_size=2000)
    dataset=dataset.batch(200)
    dataset=dataset.repeat(None)


    return dataset


#Main Function............................................................
def main(hparams):

  # Create the Estimator
  gun_detector = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir=hparams.job_dir)

  #Call to training
  gun_detector.train(
      input_fn=train_input_fn,
      steps=16000)

  val_results=gun_detector.evaluate(
      input_fn=val_input_fn)

  print(val_results)


#Arguments Function.......................................................
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--train-files',
        help='GCS or local paths to training data',
        required=True
        )

    parser.add_argument(
        '--eval-files',
        help='GCS or local paths to evaluating data',
        required=True
        )

    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        required=True
        )

    args = parser.parse_args()

    hparams=hparam.HParams(**args.__dict__)
    main(hparams)
