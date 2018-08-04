import logging

import tensorflow as tf

logger = logging.getLogger('trainer')


class DatasetBuilder(object):

    @staticmethod
    def _resize_fn(images, image_size):
        return tf.image.resize_images(
            images,
            image_size,
            method=tf.image.ResizeMethod.BICUBIC
        )

    @staticmethod
    def _decode_example(example_proto):
        features = {
            "image/encoded": tf.FixedLenFeature(
                (), tf.string, default_value=""
            )
        }
        parsed_features = tf.parse_single_example(example_proto, features)
        image = tf.image.decode_jpeg(parsed_features["image/encoded"])
        return image

    @classmethod
    def build(cls, filename, batch_size, image_size, num_epocs=None):
        logger.info('Creating dataset from: %s' % filename)
        dataset = tf.data.TFRecordDataset(filename)
        dataset = dataset.map(cls._decode_example)
        dataset = dataset.map(lambda x: cls._resize_fn(x, image_size))
        dataset = dataset.repeat(num_epocs)
        dataset = dataset.batch(batch_size)

        iterator = dataset.make_one_shot_iterator()
        return iterator
