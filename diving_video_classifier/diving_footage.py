import io

import tensorflow as tf
import PIL.Image
from utils import tfrecord_helpers
from utils import tfexample_helpers
from utils import dataset_utils


class DivingTFRecord(tfrecord_helpers.TFRecordHelper):

    def __init__(self, filename):
        super().__init__(filename)

    def build_tf_dataset(self, image_size, batch_size):
        dataset = tf.data.TFRecordDataset(self.filename)
        dataset = dataset.map(self.decode_tensor)
        dataset = dataset.map(
            lambda x: dataset_utils.resize(x, image_size)
        )
        dataset = dataset.map(dataset_utils.mobilenet_preprocess)
        dataset = dataset.batch(batch_size)

        return dataset

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
        filename = feature_dict['image/filename'].bytes_list.value[0]
        height = feature_dict['image/height'].int64_list.value[0]
        width = feature_dict['image/width'].int64_list.value[0]
        classification = (
            feature_dict['image/classification'].bytes_list.value[0]
        )

        return {
            'image': PIL.Image.open(io.BytesIO(image_value)),
            'height': height,
            'width': width,
            'filename': filename,
            'format': feature_dict['image/format'].bytes_list.value[0],
            'classification': classification,
        }

    @classmethod
    def build_example(cls, image, filename, frame_id, classification):
        image_data = tfrecord_helpers.get_jpeg_string(image)

        to_bytes = tfexample_helpers.bytes_list_feature
        to_int64 = tfexample_helpers.int64_list_feature

        return tf.train.Example(
            features=tf.train.Features(
                feature={
                    'image/encoded': to_bytes(image_data),
                    'image/filename': to_bytes(filename),
                    'image/format': to_bytes(tfexample_helpers.IMAGE_FORMAT),
                    'image/height': to_int64(image.size[0]),
                    'image/width': to_int64(image.size[1]),
                    'image/channels': to_int64(3),
                    'image/frame_id': to_int64(frame_id),
                    'image/classification': to_bytes(classification),
                }
            )
        )
