# load train, valid, test data
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from PIL import Image
from tensorflow.data import Dataset
from tensorflow.keras import datasets, models

WORKING_PATH = "./"
TEXT_LENGTH = 75
TEXT_HIDDEN = 256


class my_data_set(Dataset):
    def __init__(self, path, text_length):
        super(my_data_set, self).__init__()
        self.working_path = path
        self.text_length = text_length

        # the index of words in the twitter text
        self.word2idx = self.__load_word2idx_dict()

        # self.img2attr: the words of the attributes of images
        # self.attr2idx: the index of words in the attributes of images
        self.img2attr, self.attr2idx = self.__load_image_attribute_dict()

        # load train, valid, test data of twitter dataset
        self.twitter_data = self.__load_twitter_data()
        self.__convert_text2idx()
        self.__convert_attr2idx()
        self.__load_image_attribute_vectors()

    def __load_word2idx_dict(self):
        working_path = self.working_path
        dict_file = open(
            os.path.join(working_path, "text_embedding", "vocab.pickle"), "rb"
        )
        word2idx = pickle.load(dict_file, encoding="utf-8")
        return word2idx

    def __load_image_attribute_dict(self):
        working_path = self.working_path
        img2attr = dict()
        with open(
            os.path.join(working_path, "image_attribute", "five_words_of_images.txt"),
            "rb",
        ) as fin:
            lines = fin.readlines()
        for line in lines:
            content = eval(line)
            img2attr[int(content[0])] = content[1:]

        attr_dict_file = open(
            os.path.join(working_path, "image_attribute_embedding", "vocab.pickle"),
            "rb",
        )
        attr2idx = pickle.load(attr_dict_file, encoding="utf-8")

        return img2attr, attr2idx

    def __load_twitter_data(self):
        """load train, valid, test data and return a dict

        Returns:
            [dict]: a dict of dataset. 
            key is the index of image and the content is {'text': ..., 'label':...}
        """
        data_set = dict()
        working_path = self.working_path
        # load the train data
        for dataset in ["train"]:
            with open(
                os.path.join(working_path, "text_data", dataset + ".txt"), "rb"
            ) as fin:
                lines = fin.readlines()
            for line in lines:
                content = eval(line)
                image_idx, sentence, label = content[0], content[1], content[2]
                image_path = os.path.join(
                    working_path, "image_data", image_idx + ".jpg"
                )
                if os.path.isfile(image_path):
                    data_set[int(image_idx)] = {
                        "text": sentence,
                        "label": label,
                        "image_path": image_path,
                    }

        # load valid and test data
        for dataset in ["valid", "test"]:
            with open(
                os.path.join(working_path, "text_data", dataset + ".txt"), "rb"
            ) as fin:
                lines = fin.readlines()
            for line in lines:
                content = eval(line)
                image_idx, sentence, real_label = (
                    content[0],
                    content[1],
                    content[3],
                )  # NOTE: use content[3] (manual labeling) as the real label
                image_path = os.path.join(
                    working_path, "image_data", image_idx + ".jpg"
                )
                if os.path.isfile(image_path):
                    data_set[int(image_idx)] = {
                        "text": sentence,
                        "label": real_label,
                        "image_path": image_path,
                    }

        return data_set

    def __convert_text2idx(self):
        # self.image_idx = list(self.twitter_data.keys())
        for idx in self.twitter_data.keys():
            text = self.twitter_data[idx]["text"].split()
            indexed_text = tf.zeros([self.text_length], dtype=tf.int32)
            for i, word in enumerate(text):
                if i >= self.text_length:
                    break
                indexed_text[i] = self.word2idx.get(word, self.word2idx["<unk>"])
            self.twitter_data[idx]["indexed_text"] = indexed_text

    def __convert_attr2idx(self):
        

def test():
    dataset = my_data_set(os.getcwd())


if __name__ == "__main__":
    test()
