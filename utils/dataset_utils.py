import tensorflow as tf


def resize(example, image_size, image_key='image'):
    """Resizes image

    """
    example[image_key] = tf.image.resize_images(
        example[image_key],
        image_size,
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    return example


def mobilenet_preprocess(example, image_key='image'):
    example[image_key] = tf.cast(example[image_key], tf.float32) / 128. - 1.
    return example
