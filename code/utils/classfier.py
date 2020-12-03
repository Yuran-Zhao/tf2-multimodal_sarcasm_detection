# 2 fc layer as classifer
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class ClassificationLayer(tf.Module):
    def __init__(self, name=None):
        super(ClassificationLayer, self).__init__(name=name)
        self.Linear_1 = layers.Dense(256, input_shape=(512,))
        self.Linear_2 = layers.Dense(1, input_shape=(256,), activation="sigmoid")

    @tf.function()
    def __call__(self, x):
        hidden = self.Linear_1(x)
        output = self.Linear_2(hidden)
        return output


def test():
    print("=" * 30)
    print("Begin testing ClassificationLayer...")
    print("=" * 30)
    cls = ClassificationLayer()
    batch_size = 50
    a = np.random.randn(batch_size, 512)
    a = tf.convert_to_tensor(a, dtype=tf.float32)
    output = cls(a)
    assert list(output.shape) == list((50, 1))
    print("Pass the test!!!")


if __name__ == "__main__":
    test()
