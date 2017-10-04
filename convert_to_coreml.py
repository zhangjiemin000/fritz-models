import argparse
import keras_contrib
import logging

import layer_converters
import layers
import models
from fritz_coreml_converter import FritzCoremlConverter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('stylize_image')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Stylize an image using a trained model.'
    )

    parser.add_argument(
        '--weights-checkpoint', type=str, required=True,
        help='Weights from a trained Style Transfer Network.'
    )
    parser.add_argument(
        '--coreml-model', type=str, required=True,
        help='A CoreML output file to save to'
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

    # Map custom layers to their custom coreml converters
    custom_layers = {
        keras_contrib.layers.normalization.InstanceNormalization: layer_converters.convert_instancenormalization,  # NOQA
        layers.DeprocessStylizedImage: layer_converters.convert_deprocessstylizedimage  # NOQA
    }

    logger.info('Loading model weights from %s' % args.weights_checkpoint)
    transfer_net = models.StyleTransferNetwork.build(
        args.img_height, args.img_width
    )
    transfer_net.load_weights(args.weights_checkpoint)

    fritz_converter = FritzCoremlConverter()
    mlmodel = fritz_converter.convert_keras(
        transfer_net,
        custom_layers=custom_layers,
        image_input_names=['input1'],
        image_output_names=['output1'],
        deprocessing_args={
            'is_bgr': False,
            'image_scale': 127.5,
            'red_bias': 127.5,
            'green_bias': 127.5,
            'blue_bias': 127.5
        }
    )
    logger.info('Saving .mlmodel to %s' % args.coreml_model)
    mlmodel.save(args.coreml_model)
