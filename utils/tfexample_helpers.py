import collections
import tensorflow as tf
IMAGE_FORMAT = 'jpeg'
LABEL_FORMAT = 'png'


def int64_list_feature(values):
    """Return a TF-Feature of int64_list.

    Args:
        values: A scalar or list of values.
    Returns:
        A TF-Feature.
    """
    if not isinstance(values, collections.Iterable):
        values = [values]

    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def float64_list_feature(values):
    """Return a TF-Feature of float64_list.

    Args:
        values: A scalar or list of values.
    Returns:
        A TF-Feature.
    """
    if not isinstance(values, collections.Iterable):
        values = [values]

    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def bytes_list_feature(values, multiple=False):
    """Return a TF-Feature of bytes.

    Args:
        values: A string.
    Returns:
        A TF-Feature.
    """
    def norm2bytes(value):
        return value.encode() if isinstance(value, str)

    if multiple:
        return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[norm2bytes(value)
                                                 for value in values]))
    else:
        return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[norm2bytes(values)]))
