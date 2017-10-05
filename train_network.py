import argparse
import logging
import os

import trainer
import models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('train_network')

# The default layers are those suggested by Johnson et al.
# The names map to those used in the VGG16 application included
# with Keras.
_DEFAULT_STYLE_LAYERS = [
    'block1_conv2', 'block2_conv2',
    'block3_conv3', 'block4_conv3'
]
_DEFAULT_CONTENT_LAYERS = ['block3_conv3']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a Style Transfer Network.'
    )

    parser.add_argument(
        '--training-image-dset', type=str, required=True,
        help=('An h5 file containing images to trian with. The dset must '
              'contain a key `images` with the arrays.')
    )
    parser.add_argument(
        '--style-images', type=str, required=True,
        help='A comma separated list of images to take styles from.'
    )
    parser.add_argument(
        '--weights-checkpoint', type=str, required=True,
        help='An file to save the trained network weights to.'
    )
    parser.add_argument(
        '--img-height', default=256, type=int,
        help='The height of training images.'
    )
    parser.add_argument(
        '--img-width', default=256, type=int,
        help='The width of training images.'
    )
    parser.add_argument(
        '--content-layers', type=str,
        help=('A comma separated list of VGG layers to use for '
              'computing content loss')
    )
    parser.add_argument(
        '--style-layers', type=str,
        help=('A comma separated list of VGG layers to use for '
              'computing style loss')
    )
    parser.add_argument(
        '--content-weight', type=float, default=1.0,
        help='Content loss weight'
    )
    parser.add_argument(
        '--style-weight', type=float, default=10.0,
        help='Style loss weight'
    )
    parser.add_argument(
        '--total-variation-weight', type=float, default=1e-3,
        help='Total variation loss weight'
    )
    parser.add_argument(
        '--num-iterations', type=int, default=40000,
        help='Number of iterations to train for.'
    )
    parser.add_argument(
        '--batch-size', type=int, default=4,
        help='The batch size to train with.'
    )
    parser.add_argument(
        '--learning-rate', type=float, default=0.001,
        help='The learning rate.'
    )
    parser.add_argument(
        '--log-interval', type=int, default=10,
        help='the interval at which log statements are printed.'
    )
    parser.add_argument(
        '--checkpoint-interval', type=int, default=10,
        help='the interval at which model checkpoints are saved.'
    )
    parser.add_argument(
        '--fine-tune', action='store_true',
        help=('When present, attempt to load model weights from the checkpoint'
              'file and resume training from there. If no file exists, weights'
              ' are initialized randomly.')
    )

    args = parser.parse_args()

    # Set the content and style loss layers.
    content_layers = _DEFAULT_CONTENT_LAYERS
    if args.content_layers:
        content_layers = args.content_layers.split(',')

    style_layers = _DEFAULT_STYLE_LAYERS
    if args.style_layers:
        style_layers = args.style_layers.split(',')

    style_image_files = args.style_images.split(',')

    # Create the Style Transfer Network to train.
    transfer_net = models.StyleTransferNetwork.build(
        args.img_height, args.img_width
    )
    if args.fine_tune and os.path.exists(args.weights_checkpoint):
        logger.info('Loading model weights from %s' % args.weights_checkpoint)
        transfer_net.load_weights(args.weights_checkpoint)

    model_trainer = trainer.Trainer(transfer_net)
    model_trainer.train(
        args.training_image_dset,
        style_image_files,
        args.weights_checkpoint,
        content_layers,
        style_layers,
        content_weight=args.content_weight,
        style_weight=args.style_weight,
        total_variation_weight=args.total_variation_weight,
        img_height=args.img_height,
        img_width=args.img_width,
        batch_size=args.batch_size,
        num_iterations=args.num_iterations,
        learning_rate=args.learning_rate,
        log_interval=args.log_interval,
        checkpoint_interval=args.checkpoint_interval
    )
    logger.info('Done.')
