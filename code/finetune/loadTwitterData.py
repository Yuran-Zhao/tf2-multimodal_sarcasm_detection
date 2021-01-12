import glob
import os
import pathlib
import pickle
from collections import defaultdict

import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tqdm import tqdm


def get_all_images():
    data_root = "/data4/zyr/projects/HierachicalFusionModel/data/image_data"
    data_root = pathlib.Path(data_root)
    all_image_paths = list(data_root.glob("*.jpg"))
    all_image_paths = [str(path) for path in all_image_paths]
    return all_image_paths


def get_dataset():
    print("Start Loading the Twitter dataset...")
    all_image_paths = get_all_images()

    def prepocess_image(image_raw):
        image = tf.convert_to_tensor(image_raw)
        image = tf.cast(image, tf.float32)
        image = image / 255
        return image

    def load_and_preprocess_image(image_file):
        image_file = image_file.numpy().decode("utf-8")
        keras_image = img_to_array(
            load_img(image_file, target_size=(224, 224), interpolation="bilinear")
        )
        return prepocess_image(keras_image)

    def load_and_preprocess_image_wapper(image_file):
        return tf.py_function(
            load_and_preprocess_image, inp=[image_file], Tout=tf.float32
        )

    def parse_idx(image_path):
        image_path = image_path.numpy().decode("utf-8")
        image_idx = os.path.splitext(os.path.split(image_path)[-1])[0]
        image_idx = tf.cast(image_idx, tf.string)
        return image_idx

    def parse_idx_wrapper(image_path):
        return tf.py_function(parse_idx, inp=[image_path], Tout=tf.string)

    image_path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = image_path_ds.map(
        load_and_preprocess_image_wapper,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    idx_ds = image_path_ds.map(
        parse_idx_wrapper, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    ds = tf.data.Dataset.zip((image_ds, idx_ds))
    return ds

    # img_tensor_dict = defaultdict()
    # for image_path in tqdm(all_image_paths):
    # image_idx_list = list(img_tensor_dict.keys())
    # image_tensor_list = [img_tensor_dict[idx] for idx in image_idx_list]
    # return image_idx_list, image_tensor_list

