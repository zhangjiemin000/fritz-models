import argparse
import keras
import logging
import numpy
import PIL.Image

from style_transfer import models
from style_transfer import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('stylize_image')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Stylize an image using a trained model.'
    )

    parser.add_argument(
        '--input-image', type=str, required=True,
        help='An image to stylize.'
    )
    parser.add_argument(
        '--output-image', type=str, required=True,
        help='An output file for the stylized image.'
    )
    parser.add_argument(
        '--model-checkpoint', type=str, required=True,
        help='Checkpoint from a trained Style Transfer Network.'
    )
    parser.add_argument(
        '--img-height', default=256, type=int,
        help='The height of training images.'
    )
    parser.add_argument(
        '--img-width', default=256, type=int,
        help='The width of training images.'
    )

    args = parser.parse_args()

    logger.info('Loading model from %s' % args.model_checkpoint)
    transfer_net = models.StyleTransferNetwork.build(
        args.img_height, args.img_width
    )
    transfer_net.load_weights(args.model_checkpoint)

    inputs = [transfer_net.input, keras.backend.learning_phase()]
    outputs = [transfer_net.output]

    transfer_style = keras.backend.function(inputs, outputs)

    input_image = utils.load_image(
        args.input_image,
        args.img_height,
        args.img_width,
        expand_dims=True
    )
    output_image = transfer_style([input_image, 1])[0]
    output_image = PIL.Image.fromarray(numpy.uint8(output_image[0]))
    output_image.save(args.output_image)
