import argparse
import keras_contrib
import logging

from style_transfer import layer_converters
from style_transfer import layers
from style_transfer import models
from style_transfer.fritz_coreml_converter import FritzCoremlConverter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('stylize_image')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Stylize an image using a trained model.'
    )
    parser.add_argument(
        '--keras-checkpoint', type=str, required=True,
        help='Weights from a trained Style Transfer Network.'
    )
    parser.add_argument(
        '--alpha', type=float, required=True,
        help='The width multiplier of the network.'
    )
    parser.add_argument(
        '--coreml-model', type=str, required=True,
        help='A CoreML output file to save to'
    )
    parser.add_argument(
        '--image-size', type=str, default='640,480',
        help='The size of input and output of the final Core ML model: H,W'
    )

    args = parser.parse_args()

    image_size = [int(dim) for dim in args.image_size.split(',')]
    # Map custom layers to their custom coreml converters
    custom_layers = {
        keras_contrib.layers.normalization.InstanceNormalization: layer_converters.convert_instancenormalization,  # NOQA
        layers.DeprocessStylizedImage: layer_converters.convert_deprocessstylizedimage  # NOQA
    }
    # Get custom layers so we can load the keras model config.
    custom_objects = {
        'InstanceNormalization': keras_contrib.layers.normalization.InstanceNormalization,  # NOQA
        'DeprocessStylizedImage': layers.DeprocessStylizedImage
    }

    logger.info('Loading model weights from %s' % args.keras_checkpoint)

    model = models.StyleTransferNetwork.build(
        image_size, alpha=args.alpha, checkpoint_file=args.keras_checkpoint)

    fritz_converter = FritzCoremlConverter()
    mlmodel = fritz_converter.convert_keras(
        model,
        input_names=['image'],
        image_input_names=['image'],
        output_names=['stylizedImage'],
        image_output_names=['stylizedImage'],
        custom_layers=custom_layers
    )
    logger.info('Saving .mlmodel to %s' % args.coreml_model)
    mlmodel.save(args.coreml_model)
