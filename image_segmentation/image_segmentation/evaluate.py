import keras
from image_segmentation.image_segmentation_records import (
    ImageSegmentationTFRecord
)
from image_segmentation.icnet import ICNetModelFactory
from matplotlib import pyplot
import skimage.filters
import numpy
import skimage.transform


class EvaluateImageSegmentationModel(object):

    def __init__(self,
                 trained_model_path,
                 tfrecord_path,
                 alpha=1.0):

        model = keras.models.load_model(trained_model_path)
        self.input_shape = model.inputs[0].shape.as_list()[1:3]
        num_classes = model.outputs[0].shape.as_list()[-1]
        self.model = ICNetModelFactory.build(
            self.input_shape[0],
            num_classes,
            alpha=alpha,
            train=False,
            weights_path=trained_model_path,
        )

        self.tfrecords = ImageSegmentationTFRecord(tfrecord_path)

    def run_prediction(self, record):
        image = record['image']
        image = image.resize(self.input_shape)
        img_data = numpy.array(image)
        img_data = img_data * 1. / 255. - 0.5
        img_data = skimage.filters.gaussian(img_data, sigma=0.0)

        return self.model.predict(img_data[None, :, :, :])

    def generate_prediction_image(self, record, prediction):
        image = record['image'].resize(self.input_shape)
        yield plot_image_and_mask(
                numpy.array(image),
                prediction[0][0, :, :],
                reference_mask=record['mask'],
                alpha=0.9,
                small=True
            )


def plot_image_and_mask(img, mask, alpha=0.6, deprocess_func=None,
                        reference_mask=None,
                        show_original_image=True,
                        small=False):
    """Plot an image and overlays a transparent segmentation mask.

    Args:
        img (arr): the image data to plot
        mask (arr): the segmentation mask
        alpha (float, optional): the alpha value of the segmentation mask.
        small: If true, output small figure

    Returns:
        pyplot.plot: a plot
    """
    max_mask = numpy.argmax(mask, axis=-1)

    rows, columns = 1, 1
    if show_original_image:
        columns += 1
    if reference_mask is not None:
        columns += 1

    fig = pyplot.figure()

    if deprocess_func:
        img = deprocess_func(img)

    # Add Results plot
    column_index = 1
    fig.add_subplot(rows, columns, column_index)

    pyplot.imshow(img.astype(int))
    pyplot.imshow(
        skimage.transform.resize(
            max_mask,
            img.shape[:2],
            order=0),
        alpha=alpha)

    if reference_mask is not None:
        column_index += 1
        fig.add_subplot(rows, columns, column_index)
        pyplot.imshow(img.astype(int))
        pyplot.imshow(
            skimage.transform.resize(
                reference_mask[:, :, 0],
                img.shape[:2],
                order=0),
            alpha=alpha)

    if show_original_image:
        column_index += 1
        fig.add_subplot(rows, columns, column_index)
        pyplot.imshow(img.astype('uint8'))

    if small:
        fig.set_size_inches(columns * 5, 5)
    else:
        fig.set_size_inches(columns * 10, 10)

    return fig
