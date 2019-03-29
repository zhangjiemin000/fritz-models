import io
from image_segmentation import build_data
import random
import six
import tensorflow as tf
import numpy
import PIL


def iterate_tfrecord(filename, decode=False):
    """Iterate through a tfrecord file.

    Args:
        filename (str): Filename to iterate.
        decode (bool): Optionally pass all records to example decoder function.
            False by default.

    Returns: Iterator of tfrecords.
    """
    for record in tf.python_io.tf_record_iterator(filename):
        example = tf.train.Example()
        example.ParseFromString(record)
        if decode:
            yield decode_image_tensor(example)
        else:
            yield example


def save_tfrecords(records, output_filename):
    """Save all tfrecord examples to file.

    Args:
        records (Iterator[tf.train.Example]): Iterator of records to save.
        output_filename (str): Output file to save to.
    """
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
        for record in records:
            tfrecord_writer.write(record.SerializeToString())


def decode_image_tensor(example):
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


def get_png_string(mask_array):
    """Builds PNG string from mask array.

    Args:
        mask_array (HxW): Mask array to generate PNG string from.

    Returns: String of mask encoded as a PNG.
    """
    # Convert the new mask back to an image.
    image = PIL.Image.fromarray(mask_array.astype('uint8')).convert('RGB')
    # Save the new image to a PNG byte string.
    byte_buffer = io.BytesIO()
    image.save(byte_buffer, format='png')
    byte_buffer.seek(0)
    return byte_buffer.read()


def get_jpeg_string(image):
    """Builds PNG string from mask array.

    Args:
        image (PIL.Image): Mask array to generate PNG string from.

    Returns: String of mask encoded as a PNG.
    """
    # Save the new image to a PNG byte string.
    byte_buffer = io.BytesIO()
    image.save(byte_buffer, format='jpeg')
    byte_buffer.seek(0)
    return byte_buffer.read()


def build_example(filename, image, mask):
    image_str = get_jpeg_string(image)
    mask_str = get_png_string(mask)
    return build_data.image_seg_to_tfexample(image_str,
                                             filename,
                                             image.size[0],
                                             image.size[1],
                                             mask_str)


def update_mask(record, mask_array):
    """Update mask in tensorflow example.

    Args:
        record (tf.train.Example): Record to update
        mask_array (numpy.Array): HxW array of class values.

    Returns: Updated tf.train.Example.
    """
    def norm2bytes(value):
        return value.encode() if isinstance(value, str) and six.PY3 else value

    mask_data = get_png_string(mask_array)
    feature = record.features.feature['image/segmentation/class/encoded']
    feature.bytes_list.value.pop()
    feature.bytes_list.value.append(norm2bytes(mask_data))
    return record


def get_mask_ratio(example):
    total_people_pixels = example['mask'][:, :, 0].sum(axis=None)
    return total_people_pixels / (example['height'] * example['width'])


def iter_interleave(kaggle, ade20k, coco):
    """
    A generator that interleaves the output from a one or more iterators
    until they are *all* exhausted.

    """
    kaggle_finished = False
    ade20k_finished = False
    coco_finished = False
    a, b, c = 0, 0, 0

    while (not kaggle_finished) or (not ade20k_finished) or (not coco_finished):
        if not kaggle_finished:
            try:
                item = kaggle.next()
                a += 1
                if random.choice([False, True, True]):
                    yield item
            except StopIteration:
                print("kaggle finished")
                kaggle_finished = True
        if not ade20k_finished:
            try:
                item = ade20k.next()
                b += 1
                yield item
            except StopIteration:
                print("ade20k finished")
                ade20k_finished = True

        if not coco_finished:
            try:
                for _ in range(4):
                    item = coco.next()
                    c += 1
                    yield item
            except StopIteration:
                print("coco finished")
                coco_finished = True

    print(a, b, c)


def chunk_records(filename, n, start=0):
    records = iterate_tfrecord(filename)
    while True:
        for i in range(start):
            continue

        try:
            yield [records.next() for _ in range(n)]
        except StopIteration:
            return


class TFRecordHelper(object):

    def __init__(self, filename):
        self.filename = filename
        self._generator = None

    def __iter__(self):
        self._generator = self.read()
        return self

    def __next__(self):
        return next(self._generator)

    def read(self, decode=False):
        for record in iterate_tfrecord(self.filename):
            if decode:
                yield self.decode_single_example(record)
            else:
                yield record

    def write(self, records):
        save_tfrecords(records, self.filename)

    @classmethod
    def decode_tensor(cls, example):
        """Decodes a tensor, used when creating dataset for training.

        Arguments:
            example (what type????): example tensor
        Returns: Dict[str, tf.Tensor (CHANGE)]
        """
        raise NotImplementedError()

    @classmethod
    def decode_single_example(cls, example):
        """Decode example to dictionary for easy interactive use.
        """
        raise NotImplementedError()

    @classmethod
    def build_example(cls, *args):
        """Build tfrecord example from input data.

        Used to create initial dataset.
        """
        raise NotImplementedError()
