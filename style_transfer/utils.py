import io

import PIL.Image
import numpy
from tensorflow.python.lib.io import file_io


def load_image(
        filename,
        height,
        width,
        expand_dims=False):
    """Load an image and transform it to a specific size.

    Optionally, preprocess the image through the VGG preprocessor.

    Args:
        filename - an image file to load
        height - the height of the transformed image
        width - the width of the transformed image
        vgg_preprocess - if True, preprocess the image for a VGG network.
        expand_dims - Add an addition dimension (B, H, W, C), useful for
                      feeding models.
    Returns:
        img - a numpy array representing the image.
    """
    img = file_io.read_file_to_string(filename, binary_mode=True)
    img = PIL.Image.open(io.BytesIO(img))
    img = img.resize((width, height), resample=PIL.Image.BILINEAR)
    img = numpy.array(img)

    if expand_dims:
        img = numpy.expand_dims(img, axis=0)

    return img
