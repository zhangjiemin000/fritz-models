# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""Contains common utility functions and classes for building dataset.

This script contains utility functions and classes to converts dataset to
TFRecord file format with Example protos.
The Example proto contains the following fields:
    image/encoded: encoded image content.
    image/filename: image filename.
    image/format: image file format.
    image/height: image height.
    image/width: image width.
    image/channels: image channels.
    image/segmentation/class/encoded: encoded semantic segmentation content.
    image/segmentation/class/format: semantic segmentation file format.
"""
import collections
import six
import tensorflow as tf
from utils import tfrecord_helpers
import PIL
import io
IMAGE_FORMAT = 'jpeg'
LABEL_FORMAT = 'png'


def _int64_list_feature(values):
    """Return a TF-Feature of int64_list.

    Args:
        values: A scalar or list of values.
    Returns:
        A TF-Feature.
    """
    if not isinstance(values, collections.Iterable):
        values = [values]

    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _float64_list_feature(values):
    """Return a TF-Feature of float64_list.

    Args:
        values: A scalar or list of values.
    Returns:
        A TF-Feature.
    """
    if not isinstance(values, collections.Iterable):
        values = [values]

    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def _bytes_list_feature(values, multiple=False):
    """Return a TF-Feature of bytes.

    Args:
        values: A string.
    Returns:
        A TF-Feature.
    """
    def norm2bytes(value):
        return value.encode() if isinstance(value, str) and six.PY3 else value

    if multiple:
        return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[norm2bytes(value)
                                                 for value in values]))
    else:
        return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[norm2bytes(values)]))


def classified_image_to_tfexample(image, filename, frame_id, classification):
    """Convert one image/segmentation pair to tf example.

    Args:
        image_data: string of image data.
        filename: image filename.
        height: image height.
        width: image width.
        seg_data: string of semantic segmentation data.
    Returns:
        tf example of one image/segmentation pair.
    """
    image_data = tfrecord_helpers.get_jpeg_string(image)
    return tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/encoded': _bytes_list_feature(image_data),
                'image/filename': _bytes_list_feature(filename),
                'image/format': _bytes_list_feature(IMAGE_FORMAT),
                'image/height': _int64_list_feature(image.size[0]),
                'image/width': _int64_list_feature(image.size[1]),
                'image/channels': _int64_list_feature(3),
                'image/frame_id': _int64_list_feature(frame_id),
                'image/classification': _bytes_list_feature(classification),
            }
        )
    )


def predicted_images_to_tfexample(combined_results, classifications):
    """Convert one image/segmentation pair to tf example.

    Args:
        image_data: string of image data.
        filename: image filename.
        height: image height.
        width: image width.
        seg_data: string of semantic segmentation data.
    Returns:
        tf example of one image/segmentation pair.
    """
    run_length = combined_results.shape[0]
    flattened = combined_results.flatten()

    return tf.train.Example(
        features=tf.train.Features(
            feature={
                'run_length': _int64_list_feature(run_length),
                'input_data': _float64_list_feature(flattened),
                'image/classification': _int64_list_feature(
                    classifications),
            }
        )
    )
