import os
import pickle
from collections import defaultdict

import tensorflow as tf
from tqdm import tqdm

from loadCocoData import get_attributes_list
from loadTwitterData import get_dataset

BATCH_SIZE = 32


def inference(model, dataset):
    attr_list = get_attributes_list(
        os.path.join(os.getcwd(), "finetune", "attribute_list.pickle")
    )
    attr2idx = {word: idx for idx, word in enumerate(attr_list)}
    idx2attr = {idx: word for idx, word in enumerate(attr_list)}

    dataset = (
        dataset.shuffle(buffer_size=10000)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    def get_attribute(prediction):
        _, indices = tf.math.top_k(prediction, k=5, sorted=True)
        indices = indices.numpy()
        attributes = [str(idx2attr[idx]) for idx in indices]
        return attributes, indices

    attr_text_dict = defaultdict()
    attr_idx_dict = defaultdict()
    print("Start infering the twitter dataset attributes...")
    for features, idxs in tqdm(dataset):
        predictions = model(features)
        idxs = idxs.numpy()
        for batch in range(BATCH_SIZE):
            attributes, indices = get_attribute(predictions[batch])
            idx = idxs[batch].decode("utf-8")
            attr_idx_dict[idx] = indices
            attr_text_dict[idx] = attributes
    attr_idx_file_name = os.path.join(os.getcwd(), "twitter_attr_idx.pickle")
    attr_idx_file = open(attr_idx_file_name, "wb")
    pickle.dump(attr_idx_dict, attr_idx_file)

    attr_text_file_name = os.path.join(os.getcwd(), "twitter_attr_text.pickle")
    attr_text_file = open(attr_text_file_name, "wb")
    pickle.dump(attr_text_dict, attr_text_file)

    attr_text_list = [
        [str(idx)] + attr_text_dict[idx] for idx in sorted(attr_text_dict.keys())
    ]
    with open("twitter_attr_text.txt", "w", encoding="utf-8") as fout:
        for line in attr_text_list:
            fout.write(str(line) + "\n")


def main(model_file):
    dataset = get_dataset()
    model = tf.keras.models.load_model(model_file)
    inference(model, dataset)


if __name__ == "__main__":
    main("best.h5")
