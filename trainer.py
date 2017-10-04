import h5py
import keras
import logging
import numpy
import time

import models
import layers

logger = logging.getLogger('trainer')


class Trainer(object):
    """A style transfer model trainer."""

    _log_statement = '''
Iteration: {iteration} / {num_iterations}
Batch: {batch_idx} / {num_batches}
Batch Duration: {duration}s
Total Loss: {total_loss}
Style Loss: {style_loss}
Content Loss: {content_loss}
Total Variantion Loss: {total_variation_loss}
'''

    def __init__(self, transfer_net):
        """Initialize the trainer.

        The trainer is initialized with a network. This could be a brand
        new network or one that has already been trained and is going to
        be used for fine-tuning.

        Args:
            transfer_net: A functional Keras model to train.
        """
        self.transfer_net = transfer_net

    @classmethod
    def get_gram_matrix(cls, x, norm_by_channels=False):
        """Compute the Gram matrix of the tensor x.

        This code was adopted from @robertomest
        https://github.com/robertomest/neural-style-keras/blob/master/training.py  # NOQA

        Args:
            x - a tensor
            norm_by_channels - if True, normalize the Gram Matrix by the number
            of channels.
        Returns:
            gram - a tensor representing the Gram Matrix of x
        """
        if keras.backend.ndim(x) == 3:
            features = keras.backend.batch_flatten(
                keras.backend.permute_dimensions(x, (2, 0, 1))
            )

            shape = keras.backend.shape(x)
            C, H, W = shape[0], shape[1], shape[2]

            gram = keras.backend.dot(
                features,
                keras.backend.transpose(features)
            )
        elif keras.backend.ndim(x) == 4:
            # Swap from (H, W, C) to (B, C, H, W)
            x = keras.backend.permute_dimensions(x, (0, 3, 1, 2))
            shape = keras.backend.shape(x)
            B, C, H, W = shape[0], shape[1], shape[2], shape[3]

            # Reshape as a batch of 2D matrices with vectorized channels
            features = keras.backend.reshape(
                x, keras.backend.stack([B, C, H * W])
            )
            # This is a batch of Gram matrices (B, C, C).
            gram = keras.backend.batch_dot(features, features, axes=2)
        else:
            raise ValueError(
                'The input tensor should be either a 3d (H, W, C) '
                'or 4d (B, H, W, C) tensor.'
            )
        # Normalize the Gram matrix
        if norm_by_channels:
            denominator = C * H * W  # Normalization from Johnson
        else:
            denominator = H * W  # Normalization from Google
        gram = gram / keras.backend.cast(denominator, x.dtype)

        return gram

    @classmethod
    def get_content_loss(
            cls,
            transfer_image_outputs,
            original_image_outputs,
            layer_names):
        """Get content loss for each content layer.

        Args:
            transfer_image_outputs - the output at each layer of VGG16 for
                the stylized image
            original_image_outputs - the outputs at each layer of the VGG16
                for the original image
            layer_names - a list of layers to use for computing content loss

        Returns:
            content_loss - a list of content loss values for each content layer
        """
        return [
            cls.get_content_layer_loss(
                transfer_image_outputs[layer_name],
                original_image_outputs[layer_name]
            )
            for layer_name in layer_names
        ]

    @classmethod
    def get_content_layer_loss(cls, transfer_output, original_output):
        """Get content loss from a single content layer.

        Loss is defined as the L2 norm between the features of the
        original image and the stylized image.

        Args:
            transfer_output - an output tensor from a content layer
            original_output - an output tensor from a content layer

        Returns:
            loss - the content loss between the two layers
        """
        diffs = transfer_output - original_output
        return cls.get_l2_norm_loss(diffs)

    @classmethod
    def get_l2_norm_loss(cls, diffs):
        """Compute the l2 norm of diffs between layers.

        Args:
            diff - a tensor to compute the norm of

        Returns:
            norm - the L2 norm of the differences
        """
        axis = (1, 2, 3)
        if keras.backend.ndim(diffs) == 3:
            axis = (1, 2)

        return keras.backend.mean(
            keras.backend.square(diffs),
            axis=axis
        )

    @classmethod
    def get_style_loss(
            cls,
            transfer_image_outputs,
            style_image_outputs,
            layer_names,
            norm_by_channels=False):
        """Get style loss for each style layer.

        Args:
            transfer_image_outputs - the output at each layer of VGG16 for
                the stylized image
            style_image_outputs - the outputs at each layer of the VGG16
                for the image style is being transfered from
            layer_names - a list of layers to use for computing style loss
            norm_by_channel - If True, normalize Gram Matrices by channel

        Returns:
            loss - a list of content loss values for each style layer
        """
        return [
            cls.get_style_layer_loss(
                transfer_image_outputs[layer_name],
                style_image_outputs[layer_name],
                norm_by_channels=norm_by_channels
            )
            for layer_name in layer_names
        ]

    @classmethod
    def get_style_layer_loss(
            cls, transfer_output, style_output, norm_by_channels=False):
        """Get style loss from a single content layer.

        Loss is defined as the L2 norm between the Gram Matrix of features
        between the stylized image and the original artistic style image.

        Args:
            transfer_output - an output tensor from a style layer
            style_output - an output tensor from a style layer

        Returns:
            loss - the style loss between the two layers
        """
        # TODO: We could improve efficiency by precomputing the Gram Matrices
        # for the style image as they remain the same for each image.
        style_gram = cls.get_gram_matrix(
            style_output, norm_by_channels=norm_by_channels)
        transfer_gram = cls.get_gram_matrix(
            transfer_output, norm_by_channels=norm_by_channels)

        diffs = style_gram - transfer_gram
        style_layer_loss = cls.get_l2_norm_loss(diffs)
        return style_layer_loss

    @classmethod
    def get_total_variation_loss(cls, output):
        """Compute the total variation loss of a tensor.

        The TV loss is a measure of how much adjacent tensor elements differ.
        A lower TV loss generally means the resulting image is smoother.

        Args:
            output - a tensor, usually representing an image.

        Returns:
            tv_loss - the total variation loss of the tensor.
        """
        width_var = keras.backend.square(
            output[:, :-1, :-1, :] - output[:, 1:, :-1, :]
        )
        height_var = keras.backend.square(
            output[:, :-1, :-1, :] - output[:, :-1, 1:, :]
        )
        return keras.backend.sum(
            keras.backend.pow(width_var + height_var, 1.25),
            axis=(1, 2, 3)
        )

    def train(
            self,
            training_image_dset,
            style_image_files,
            weights_checkpoint_file,
            content_layers,
            style_layers,
            content_weight=1.0,
            style_weight=10,
            total_variation_weight=1e-3,
            img_height=256,
            img_width=256,
            batch_size=4,
            num_iterations=5000,
            norm_by_channels=True,
            learning_rate=0.001,
            log_interval=10):
        """Train the Transfer Network.

        The training procedure consists of iterating over images in
        the COCO image training data set, transforimg the with the style
        transfer model, then computing the total loss across style and
        content layers.

        The default parameters are those suggested by Johnson et al.

        Args:
            training_image_dset - an h5 data set containing training images
            style_image_file - a list of filenames of images that style will be
                               transfered from
            weights_checkpoint_file -  a file to write final model weights
            content_layers - a list of layers used to compute content loss
            style_layers - a list of layers used to compute style loss
            content_weight - a weight factor for content loss. Default 1.0
            style_weight - a weight factor for style loss. Default 10
            total_variation_weight - a weight factor for total variation loss.
                                     default 1e-4
            img_height - the height of the input images. Default 256
            img_width - the width of the input images. Default 256
            batch_size - the batch size of inputs each iteration. Default 4
            num_iterations - the number of training iterations. Default 5000
            norm_by_channels - bool to normalize Gram Matrices by the
                               number of channels. Default True
            learning_rate - the learning rate. Default 0.001
            log_interval -- the interval at which log statements are printed.
                            Default 10 iterations.
        """
        logger.info('Setting up network for training.')
        logger.info('Content layers: %s' % ','.join(content_layers))
        logger.info('Style layers: %s' % ','.join(style_layers))

        # Build the networks we'll use for training. For those new to Keras
        # and TensorFlow, the compute model is delayed execution. We define
        # data structures and processing steps and then execute them at a
        # later time.

        # The first network will be used to extract content features from a
        # VGG16 network. Two things need to happen. First, the default Keras
        # VGG16 network expects images to be normalized so we'll add a
        # custom normalization.
        original_content_in = layers.VGGNormalize()(self.transfer_net.input)
        # Next we need to create a VGG Network and pass it the origina,
        # unstylized image.
        content_net = models.IntermediateVGG(input_tensor=original_content_in)

        # Now create a network model to extract style features.
        # The input tensor to this model will be fed data later on.
        style_net = models.IntermediateVGG()

        # Finally, stitch the transfer model together with a VGG network to
        # extract features from the stylized image.
        # Start by normalizing the image for input into the VGG model.
        vgg_in = layers.VGGNormalize()(self.transfer_net.output)

        # Now create a VGG network and pass the output of the transfer network
        # as a previous layer.
        variable_net = models.IntermediateVGG(
            prev_layer=vgg_in,
            input_tensor=self.transfer_net.input
        )

        # Compute the losses.
        # Content Loss
        content_losses = self.get_content_loss(
            variable_net.layers, content_net.layers, content_layers)
        total_content_loss = sum(content_losses)
        weighted_total_content_loss = content_weight * total_content_loss

        # Style Loss
        style_losses = self.get_style_loss(
            variable_net.layers,
            style_net.layers,
            style_layers,
            norm_by_channels
        )
        total_style_loss = sum(style_losses)
        weighted_total_style_loss = style_weight * total_style_loss

        # Total Variation Loss
        total_variation_loss = self.get_total_variation_loss(
            self.transfer_net.output
        )
        weighted_total_variation_loss = (
            total_variation_weight * total_variation_loss
        )

        # Total all losses
        total_loss = keras.backend.variable(0.0)
        total_loss = (
            weighted_total_content_loss +
            weighted_total_style_loss +
            weighted_total_variation_loss
        )

        # Setup the optimizer
        params = self.transfer_net.trainable_weights
        constraints = {}  # There are none

        # Create an optimizer and updates
        optimizer = keras.optimizers.Adam(lr=learning_rate)
        updates = optimizer.get_updates(params, constraints, total_loss)

        # Define the training function. This takes images into the
        # transfer network as well as style images.
        # Technically the learning phase here is unneeded so long as we are
        # doing InstanceNormalization and not BatchNormalization. In the latter
        # case, be careful at which values you pass when evaluating the model.
        inputs = [
            self.transfer_net.input,
            style_net.model.input,
            keras.backend.learning_phase()
        ]

        # Output all of the losses.
        outputs = [
            total_loss,
            weighted_total_content_loss,
            weighted_total_style_loss,
            weighted_total_variation_loss,
        ]

        func_train = keras.backend.function(inputs, outputs, updates)

        # Load the style images.
        logger.info('Loading style images:\n%s' % '\n'.join(style_image_files))
        style_imgs = []
        for filename in style_image_files:
            # Note no preprocessing is done while loading.
            img = self.load_image(filename, img_height, img_width)
            style_imgs.append(img)
        style_imgs = numpy.array(style_imgs)

        # Load image batches for training
        logger.info('Loading training images: \n%s' % training_image_dset)
        train_imgs = h5py.File(training_image_dset)['images']
        dataset_size = train_imgs.shape[0]
        logger.info('Dataset has %d images.' % dataset_size)

        num_batches = int(numpy.ceil(dataset_size / batch_size))
        batch_idx = 0

        iteration_ouputs = []
        for idx in range(num_iterations):
            # Time each iteration.
            start_time = time.time()

            # If we are at the end of the data set loop back around
            if batch_idx >= num_batches - 1:
                batch_idx = 0

            # Get a batch of images
            batch = train_imgs[batch_idx * batch_size:(batch_idx + 1) * batch_size]  # NOQA

            # Evaluate the network and losses.
            out = func_train([batch, style_imgs, 1.])

            # Save the outputs and increment batches.
            iteration_ouputs.append(out)
            batch_idx += 1
            end_time = time.time()

            if idx % log_interval == 0:
                logger.info(self._log_statement.format(
                    duration=(end_time - start_time),
                    iteration=(idx + 1), num_iterations=num_iterations,
                    batch_idx=batch_idx, num_batches=num_batches,
                    total_loss=out[0], content_loss=out[1],
                    style_loss=out[2], total_variation_loss=out[3])
                )
        # Save the trained weights of the transfer network model.
        logger.info('Saving weights to %s' % weights_checkpoint_file)
        self.transfer_net.save_weights(weights_checkpoint_file)
        return iteration_ouputs
