import tensorflow as tf


# 打印时间分割线
@tf.function
def printbar():
    today_ts = tf.timestamp() % (24 * 60 * 60)

    hour = tf.cast(today_ts // 3600 + 8, tf.int32) % tf.constant(24)
    minite = tf.cast((today_ts % 3600) // 60, tf.int32)
    second = tf.cast(tf.floor(today_ts % 60), tf.int32)

    def timeformat(m):
        if tf.strings.length(tf.strings.format("{}", m)) == 1:
            return tf.strings.format("0{}", m)
        else:
            return tf.strings.format("{}", m)

    timestring = tf.strings.join(
        [timeformat(hour), timeformat(minite), timeformat(second)], separator=":"
    )
    tf.print("==========" * 8 + timestring)

