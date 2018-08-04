import argparse
import keras_contrib
import logging
import keras

from style_transfer import layer_converters
from style_transfer import layers
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
        '--coreml-model', type=str, required=True,
        help='A CoreML output file to save to'
    )

    args = parser.parse_args()

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

    transfer_net = keras.models.load_model(
        args.keras_checkpoint,
        custom_objects=custom_objects)
    fritz_converter = FritzCoremlConverter()
    mlmodel = fritz_converter.convert_keras(
        transfer_net,
        custom_layers=custom_layers,
        image_input_names=['image'],
        image_output_names=['stylizedImage'],
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
