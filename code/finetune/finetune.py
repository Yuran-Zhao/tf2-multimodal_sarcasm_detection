# finetune the resnet with coco dataset
import argparse
import datetime
import os

import tensorflow as tf
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.layers import (
    AveragePooling2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    Input,
)
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Accuracy, Mean, Metric
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

from loadCocoData import *
from multiLabelAccuracy import MultiLabelAccuracy
from utils import *

BATCH_SIZE = 32
EPOCH = 5
log_dir = "runs/"
run_name = "finetune" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
train_file_writer = tf.summary.create_file_writer(log_dir + run_name + "/train")
val_file_writer = tf.summary.create_file_writer(log_dir + run_name + "/validation")


@tf.function
def train_step(model, features, labels, loss_fn, optimizer, train_loss, train_metric):
    with tf.GradientTape() as tape:
        predictions = model(features, training=True)
        loss = loss_fn(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss.update_state(loss)
    train_metric.update_state(labels, predictions)


@tf.function
def valid_step(model, features, labels, loss_fn, valid_loss, valid_metric):
    predictions = model(features)
    batch_loss = loss_fn(labels, predictions)
    valid_loss.update_state(batch_loss)
    valid_metric.update_state(labels, predictions)


def train_model(
    model,
    ds_train,
    ds_valid,
    epoches,
    train_loss,
    valid_loss,
    train_metric,
    valid_metric,
    optimizer,
    loss_fn,
    save_file_name,
):
    acc = 0.0
    for epoch in tf.range(1, epoches + 1):
        for features, labels in tqdm(ds_train):
            train_step(
                model, features, labels, loss_fn, optimizer, train_loss, train_metric
            )

        with train_file_writer.as_default():
            tf.summary.scalar(
                "epoch_loss", train_loss.result(), step=tf.cast(epoch, tf.int64)
            )
            tf.summary.scalar(
                "epoch_accuracy", train_metric.result(), step=tf.cast(epoch, tf.int64)
            )

        for features, labels in tqdm(ds_valid):
            valid_step(model, features, labels, loss_fn, valid_loss, valid_metric)

        with val_file_writer.as_default():
            tf.summary.scalar(
                "epoch_loss", valid_loss.result(), step=tf.cast(epoch, tf.int64),
            )
            tf.summary.scalar(
                "epoch_accuracy", valid_metric.result(), step=tf.cast(epoch, tf.int64),
            )

        logs = "Epoch={},Loss:{},Accuracy:{},Valid Loss:{},Valid Accuracy:{}"

        if epoch % 1 == 0:
            printbar()
            tf.print(
                tf.strings.format(
                    logs,
                    (
                        epoch,
                        train_loss.result(),
                        train_metric.result(),
                        valid_loss.result(),
                        valid_metric.result(),
                    ),
                )
            )
            tf.print("")

        if valid_metric.result() > acc:
            acc = valid_metric.result()
            model.save(save_file_name)

        train_loss.reset_states()
        valid_loss.reset_states()
        train_metric.reset_states()
        valid_metric.reset_states()


def create_model():
    base_model = ResNet101(include_top=False, weights="imagenet")
    head_model = base_model.output
    head_model = GlobalAveragePooling2D()(head_model)
    # maybe add a hidden layer will improve the model?
    # head_model = Dense(4096, activation="relu")(head_model)
    # use sigmoid activation function for multi-label task
    head_model = Dense(1000, activation="sigmoid")(head_model)
    model = Model(inputs=base_model.input, outputs=head_model)
    return base_model, model


def finetune(freeze, epoches=5, model_file=None):
    if freeze and model_file is None:
        base_model, model = create_model()
        for layer in base_model.layers:
            layer.trainable = False
    elif not freeze and model_file is not None:
        base_model = None
        model = tf.keras.models.load_model(model_file)
        for layer in model.layers:
            layer.trainable = True
    else:
        print(
            "[Error] The `freeze` and `model_file` should be consistant when calling def finetune"
        )
        return

    coco_train, coco_val, coco_test = (
        load_coco_data("train"),
        load_coco_data("val"),
        load_coco_data("test"),
    )
    coco_train = (
        coco_train.shuffle(buffer_size=10000)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    coco_val = (
        coco_val.shuffle(buffer_size=10000)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    optimizer = Adam()
    loss_fn = BinaryCrossentropy()
    train_loss = Mean(name="train_loss")
    train_metric = MultiLabelAccuracy(name="train_accuracy")
    valid_loss = Mean(name="valid_loss")
    valid_metric = MultiLabelAccuracy(name="valid_accuracy")

    if freeze:
        train_model(
            model,
            coco_train,
            coco_val,
            epoches,
            train_loss,
            valid_loss,
            train_metric,
            valid_metric,
            optimizer,
            loss_fn,
            "freeze_best.h5",
        )
    else:
        train_model(
            model,
            coco_train,
            coco_val,
            epoches,
            train_loss,
            valid_loss,
            train_metric,
            valid_metric,
            optimizer,
            loss_fn,
            "best.h5",
        )


if __name__ == "__main__":
    finetune(True, 200, None)

