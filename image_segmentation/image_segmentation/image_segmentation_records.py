import io

import tensorflow as tf
import PIL.Image
from fritz.train import tfrecord_helpers
from fritz.train import tfexample_helpers

from image_segmentation.data_generator import ADE20KDatasetBuilder

import numpy
import PIL
LABEL_FORMAT = 'png'


class ImageSegmentationTFRecord(tfrecord_helpers.TFRecordHelper):

    def build_tf_dataset(self, classes, image_size, batch_size,
                         parallel_calls=1):
        return ADE20KDatasetBuilder.build(
            self.filenames,
            n_classes=len(classes),
            batch_size=batch_size,
            image_size=(image_size, image_size),
            augment_images=False,
            parallel_calls=parallel_calls,
            prefetch=True,
        )

    @classmethod
    def decode_tensor(cls, example):
        features = {
            "image/encoded": tf.FixedLenFeature(
                (), tf.string, default_value=""
            ),
            "classification": tf.FixedLenFeature(
                (), tf.string, default_value=""
            )
        }

        parsed_features = tf.parse_single_example(example, features)
        image = tf.image.decode_jpeg(
            parsed_features["image/encoded"], channels=3
        )

        return {'image': image, 'label': parsed_features['classification']}

    @classmethod
    def decode_single_example(cls, example):
        """Takes a tfrecord example and decodes image and mask data.
        Args:
            example (tf.train.Example): TF example to decode.
        Returns: dict of decoded mask and image data.
        """
        feature_dict = example.features.feature
        image_value = feature_dict['image/encoded'].bytes_list.value[0]
        encoded_mask = feature_dict['image/segmentation/class/encoded']
        filename = feature_dict['image/filename'].bytes_list.value[0]
        mask_value = encoded_mask.bytes_list.value[0]
        mask = numpy.array(PIL.Image.open(io.BytesIO(mask_value)))
        height = feature_dict['image/height'].int64_list.value[0]
        width = feature_dict['image/width'].int64_list.value[0]
        mask_format = (
            feature_dict['image/segmentation/class/format'].bytes_list.value[0]
        )

        return {
            'image': PIL.Image.open(io.BytesIO(image_value)),
            'mask': mask,
            'height': height,
            'width': width,
            'filename': filename,
            'format': feature_dict['image/format'].bytes_list.value[0],
            'mask_format': mask_format,
        }

    @classmethod
    def build_example(cls, image_data, filename, height, width, seg_data):
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
        to_bytes = tfexample_helpers.bytes_list_feature
        to_int64 = tfexample_helpers.int64_list_feature

        return tf.train.Example(
            features=tf.train.Features(
                feature={
                    'image/encoded': to_bytes(image_data),
                    'image/filename': to_bytes(filename),
                    'image/format': to_bytes(tfexample_helpers.IMAGE_FORMAT),
                    'image/height': to_int64(height),
                    'image/width': to_int64(width),
                    'image/channels': to_int64(3),
                    'image/segmentation/class/encoded': (
                        to_bytes(seg_data)),
                    'image/segmentation/class/format': to_bytes(LABEL_FORMAT),
                }
            )
        )
