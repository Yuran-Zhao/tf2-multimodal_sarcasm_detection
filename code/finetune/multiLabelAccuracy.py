import tensorflow as tf
from tensorflow.keras.metrics import Metric


class MultiLabelAccuracy(Metric):
    def __init__(self, name="multilabelaccuracy", **kwargs):
        super(MultiLabelAccuracy, self).__init__(name=name, **kwargs)
        self.right_cnt = self.add_weight(name="right", shape=(1,), initializer="zeros")
        self.total_cnt = self.add_weight(name="total", shape=(1,), initializer="zeros")

    @tf.function
    def update_state(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.float32)
        batch_size = y_true.shape[0]
        _, indices = tf.math.top_k(y_pred, k=5, sorted=True)
        for batch in range(batch_size):
            for idx in indices[batch]:
                if y_true[batch][idx]:
                    self.right_cnt.assign(self.right_cnt + 1)
                self.total_cnt.assign(self.total_cnt + 1)
        return (self.right_cnt, self.total_cnt)

    @tf.function
    def result(self):
        return tf.truediv(self.right_cnt, self.total_cnt)[0]

    @tf.function
    def reset_states(self):
        self.total_cnt.assign(tf.zeros([1]))
        self.right_cnt.assign(tf.zeros([1]))
