
import tensorflow as tf
from utils import tfrecord_helpers
from utils import tfexample_helpers


class EncodedDivingTFRecord(tfrecord_helpers.TFRecordHelper):

    def __init__(self, filename):
        super().__init__(filename)

    def build_tf_dataset(self, image_size, batch_size, sequence_length):
        dataset = tf.data.TFRecordDataset(self.filename)
        dataset = dataset.shuffle(500)
        dataset = dataset.map(
            lambda x: self.decode_tensor(x, sequence_length)
        )
        dataset = dataset.repeat()

        dataset = dataset.batch(batch_size)
        return dataset

    @classmethod
    def decode_tensor(cls, example, sequence_length):
        features = {
            "input_data": tf.FixedLenFeature(
                [sequence_length * 1280],
                tf.float32
            ),
            "image/classification": tf.FixedLenFeature(
                [sequence_length], tf.int64
            )
        }
        parsed_features = tf.parse_single_example(example, features)
        data = tf.reshape(parsed_features['input_data'], [sequence_length, -1])

        return {'image': data,
                'label': parsed_features['image/classification']}

    @classmethod
    def decode_single_example(cls, example):
        """Takes a tfrecord example and decodes image and mask data.

        Args:
            example (tf.train.Example): TF example to decode.

        Returns: dict of decoded mask and image data.
        """
        raise NotImplementedError()

    @classmethod
    def build_example(cls, combined_results, classifications):
        """Convert one image/segmentation pair to tf example.

        Args:
        Returns:
            tf example of one image/segmentation pair.
        """
        run_length = combined_results.shape[0]
        flattened = combined_results.flatten()
        to_int64 = tfexample_helpers.int64_list_feature
        to_float = tfexample_helpers.float64_list_feature

        return tf.train.Example(
            features=tf.train.Features(
                feature={
                    'run_length': to_int64(run_length),
                    'input_data': to_float(flattened),
                    'image/classification': to_int64(classifications),
                }
            )
        )
