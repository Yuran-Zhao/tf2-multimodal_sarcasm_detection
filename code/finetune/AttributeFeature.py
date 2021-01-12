import os
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tqdm import tqdm

from loadCocoData import get_attributes_list


def inference(model_file, image_path):
    model = tf.keras.models.load_model(model_file)

    def prepocess_image(image_raw):
        image = tf.convert_to_tensor(image_raw)
        image = tf.cast(image, tf.float32)
        image = image / 255
        return image

    def load_and_preprocess_image(image_file):
        keras_image = img_to_array(
            load_img(image_file, target_size=(224, 224), interpolation="bilinear")
        )
        return prepocess_image(keras_image)

    image = load_and_preprocess_image(image_path)
    image_idx = os.path.splitext(os.path.split(image_path)[-1])[0]
    attr_list = get_attributes_list(
        os.path.join(os.getcwd(), "finetune", "attribute_list.pickle")
    )
    attr2idx = {word: idx for idx, word in enumerate(attr_list)}
    idx2attr = {idx: word for idx, word in enumerate(attr_list)}

    def get_attribute(prediction):
        _, indices = tf.math.top_k(prediction, k=5, sorted=True)
        indices = indices.numpy()
        attributes = [str(idx2attr[idx]) for idx in indices]
        return attributes, indices

    prediction = model(image)
    attributes, indices = get_attribute(prediction)
    return attributes, indices
