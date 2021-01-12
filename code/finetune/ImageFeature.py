import os
import pickle
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, ResNet50V2
from tensorflow.keras.layers import (
    AveragePooling2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    Input,
)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tqdm import tqdm

sub_image_size = 32


def inference_image_feature(image_file):
    base_model = ResNet50V2(include_top=False)
    head_model = base_model.output
    head_model = GlobalAveragePooling2D()(head_model)
    model = Model(inputs=base_model.input, outputs=head_model)
    for layer in base_model.layers:
        layer.trainable = False

    def prepocess_image(image_raw):
        image = tf.convert_to_tensor(image_raw)
        image = tf.cast(image, tf.float32)
        image = image / 255
        return image

    def load_and_preprocess_image(image_file):
        keras_image = img_to_array(
            load_img(image_file, target_size=(448, 448), interpolation="bilinear")
        )
        return prepocess_image(keras_image)

    def process_sub_image(image):
        image = tf.keras.preprocessing.image.smart_resize(image, (256, 256))
        mean = tf.convert_to_tensor([0.485, 0.456, 0.406], dtype=tf.float32)
        mean = tf.reshape(mean, [1, 1, 3])
        std = tf.convert_to_tensor([0.229, 0.224, 0.225], dtype=tf.float32)
        std = tf.reshape(std, [1, 1, 3])
        image = (image - mean) / std
        return image

    image = load_and_preprocess_image(image_file)
    image_output = []
    for column in range(14):
        for row in range(14):
            sub_image_raw = image[
                sub_image_size * row : sub_image_size * (row + 1),
                sub_image_size * column : sub_image_size * (column + 1),
                :,
            ]
            sub_image = process_sub_image(sub_image_raw)
            # sub_image_output = base_model(tf.expand_dims(sub_image, axis=0))
            sub_image_output = model(tf.expand_dims(sub_image, axis=0))
            image_output.append(sub_image_output.numpy())
    image_output = np.array(image_output).transpose([1, 0, 2])
    print(image_output.shape)
    return image_output


def main(image_file):
    working_path = "/data4/zyr/projects/HierachicalFusionModel/data/image_data"
    image_path = os.path.join(working_path, image_file + ".jpg")
    image = inference_image_feature(image_path)


if __name__ == "__main__":
    main("840006160660983809")
