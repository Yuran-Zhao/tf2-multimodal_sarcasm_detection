import argparse
import os
import pickle
from collections import Counter, defaultdict

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
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
from tensorflow.keras.metrics import Accuracy, Mean, Metric, TopKCategoricalAccuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

WORKING_PATH = os.getcwd()
for w in ["!", ",", ".", "?", "-s", "-ly", "</s>", "s", " "]:
    STOP_WORDS.add(w)


def write_captions_of_iamges_to_file(dataset, split):
    """write all captions of images in the dataset to a .txt.
        in case of the future usage

    Args:
        dataset (tf.data.Dataset): a coco dataset
        split (str): the split of the dataset (train, val, test)
    """
    # print("Writing captions of images into {}...".format(split + "_captions.txt"))
    write_to_file = os.path.join(WORKING_PATH, "finetune", split + "_captions.txt")
    if os.path.exists(write_to_file):
        # print("The {} has already existed...".format(split + "_captions.txt"))
        return
    captions = []
    for feature in tqdm(dataset.take(len(dataset))):
        text = feature["captions"]["text"].numpy()
        for i in range(len(text)):
            t = text[i].decode("utf-8")
            captions.append(t)
    with open(write_to_file, "w", encoding="utf-8") as fout:
        for line in captions:
            fout.write(line + "\n")


def get_captions_of_images(dataset, split):
    """get the captions of every image in the dataset, return a dict

    Args:
        dataset (tf.data.Dataset): a coco dataset
        split (str): the split of the dataset (train, val, test)

    Returns:
        defaultdict: a dict contains image_id and its corresponding caption
    """
    # print("Geting the captions of images...")
    img_cap_dict_file_name = os.path.join(
        WORKING_PATH, "finetune", split + "_img_cap_dict.pickle"
    )
    if os.path.exists(img_cap_dict_file_name):
        # print("The {} has already existed...".format(split + "_img_cap_dict.pickle"))
        img_cap_dict_file = open(img_cap_dict_file_name, "rb")
        img_cap_dict = pickle.load(img_cap_dict_file)
        return img_cap_dict
    img_cap_dict = defaultdict()
    for feature in tqdm(dataset.take(len(dataset))):
        text, image_id = (
            feature["captions"]["text"].numpy(),
            feature["image/id"].numpy(),
        )
        cur = []
        for i in range(len(text)):
            t = text[i].decode("utf-8")
            cur.append(t)
        img_cap_dict[image_id] = " ".join(cur)
    img_cap_dict_file = open(img_cap_dict_file_name, "wb")
    pickle.dump(img_cap_dict, img_cap_dict_file)
    return img_cap_dict


def get_attributes_list(file_name):
    """load the attribute_list, which is a .pickle file.

    Args:
        file_name (str): the path of attribute_list.pickle

    Returns:
        list: the attributes list
    """
    attr_file = open(file_name, "rb")
    attr_list = pickle.load(attr_file)
    return attr_list


def get_attributes_dict(dataset, split, attr_list, img_cap_dict):
    """get the top 5 attributes of every image in the dataset

    Args:
        dataset (tf.data.Dataset): a coco dataset
        split (split): the split of the dataset (train, val, test)
        attr_list (list): the target attributes list containing 1000 attributes.
        img_cap_dict (dict): the dictory contains images and its corresponding captions

    Returns:
        dict: the dictory contains image_id and its top 5 attributes
    """
    # print("Getting the attributes of images...")
    attr_dict_file_name = os.path.join(
        WORKING_PATH, "finetune", split + "_attr_dict.pickle"
    )
    if os.path.exists(attr_dict_file_name):
        # print("The {} has already existed...".format(split + "_attr_dict.pickle"))
        attr_dict_file = open(attr_dict_file_name, "rb")
        attr_dict = pickle.load(attr_dict_file)
        return attr_dict
    attr_dict = defaultdict()
    nlp = English()

    def init_attr_dict():
        init_dict = defaultdict()
        for attr in attr_list:
            init_dict[attr] = 0
        return Counter(init_dict)

    for img_id in tqdm(img_cap_dict.keys()):
        init_dict = init_attr_dict()
        text = img_cap_dict[img_id].lower()
        text = nlp(text)
        text = [token.text for token in text if token.text not in STOP_WORDS]
        for word in text:
            if word in init_dict:
                init_dict[word] += 1
        cnt = Counter(init_dict)
        attr = [word for word, _ in cnt.most_common(5)]
        attr_dict[img_id] = attr
    attr_dict_file = open(attr_dict_file_name, "wb")
    pickle.dump(attr_dict, attr_dict_file)
    return attr_dict


def get_onehot_attributes(attr_dict, attr2idx, split):
    """get the labels in onehot format

    Args:
        attr_dict (dict: the dictory contains image_id and its top 5 attributes
        attr2idx (dict): the dictory contains corresponding index of attributes
        split (str): the split of the dataset (train, val, test)

    Returns:
        dict: the dictory contains every image and its top 5 attributes
    """
    # print("Getting the onehot labels of images...")
    attr_label_file_name = os.path.join(
        WORKING_PATH, "finetune", split + "_onehot_attribute.pickle"
    )
    if os.path.exists(attr_label_file_name):
        # print(
        #     "The {} has already existed...".format(split + "_onehot_attribute.pickle")
        # )
        attr_label_file = open(attr_label_file_name, "rb")
        attr_label = pickle.load(attr_label_file)
        return attr_label
    attr_label = defaultdict()

    def generate_onehot(attr):
        onehot = [0] * 1000
        for idx in attr:
            onehot[idx] = 1
        return tf.stack(onehot)

    for img_id in tqdm(attr_dict.keys()):
        attr_index = [attr2idx[word] for word in attr_dict[img_id]]
        attr_label[img_id] = generate_onehot(attr_index)
    attr_label_file = open(attr_label_file_name, "wb")
    pickle.dump(attr_label, attr_label_file)
    return attr_label


def load_coco_data(split):
    """load the `split` data containing image and label

    Args:
        split (str): the split of the dataset (train, val, test)

    Returns:
        tf.data.Dataset: the dataset contains image and label
        image (tf.tensor), shape (224, 224, 3)
        label (tf.tensor), shape (1000, )
    """
    dataset = tfds.load(name="coco_captions", split=split)
    write_captions_of_iamges_to_file(dataset, split)
    img_cap_dict = get_captions_of_images(dataset, split)
    attr_list = get_attributes_list(
        os.path.join(WORKING_PATH, "finetune", "attribute_list.pickle")
    )
    attr2idx = {word: idx for idx, word in enumerate(attr_list)}
    attr_dict = get_attributes_dict(dataset, split, attr_list, img_cap_dict)
    attr_onehot = get_onehot_attributes(attr_dict, attr2idx, split)
    attr_onehot_labels = [attr_onehot[idx] for idx in attr_onehot.keys()]
    attr_onehot_labels = tf.data.Dataset.from_tensor_slices(
        tf.cast(attr_onehot_labels, tf.int32)
    )

    def process(image):
        image = tf.image.resize(image, (224, 224))
        image = tf.cast(image, tf.float32)
        image = image / 255
        return image

    def parse_fn(feature):
        image = feature["image"]
        return process(image)

    img_dataset = dataset.map(parse_fn)

    ds = tf.data.Dataset.zip((img_dataset, attr_onehot_labels))
    return ds
