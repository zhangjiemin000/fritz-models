import argparse
import h5py
import logging
import numpy
import os
from Queue import Queue
import sys
import threading
import urllib
import zipfile

import utils

logger = logging.getLogger('create_training_dataset')

_COCO_ZIP_URL = 'http://images.cocodataset.org/zips/train2014.zip'


class CocoPreprocessor(object):
    """A class to preprocess images from the COCO training data.

    This does not apply any sort of normalization to images. It simply
    transforms and scales image sizes before packing them into an H5 dataset
    and saving them to disk.

    Most of the code here was cribbed from @robertomest.
    """

    allowed_formats = {'.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG'}
    max_resize = 16

    @classmethod
    def process_images(
            cls,
            input_dir,
            output_filename,
            image_size,
            num_images=None,
            num_threads=1):
        """Process all images in a directory and create an H5 data set.

        Args:
            input_dir - a directory containing images
            output_filename - the name of the h5 file to write to
            image_size - a tuple (height, width) to resize images to
            num_images - the number of images to process. 'None' processes all
            num_threads - the number of threads to use. Default 1.
        """
        img_height, img_width = image_size
        img_list = cls._get_image_filenames(input_dir, num_images)

        # Remove the h5 file if it exists
        try:
            os.remove(output_filename)
        except OSError:
            pass

        h5_file = h5py.File(output_filename, 'w')
        dset_shape = (len(img_list), img_height, img_width, 3)
        imgs_dset = h5_file.create_dataset('images', shape=dset_shape)

        # input_queue stores (idx, filename) tuples,
        # output_queue stores (idx, resized_img) tuples
        input_queue = Queue()
        output_queue = Queue()

        # Read workers pull images off disk and resize them
        def read_worker(height, width, max_resize):
            while True:
                idx, filename = input_queue.get()
                try:
                    img = utils.load_image(filename, img_height, img_width)
                except (ValueError, IndexError) as e:
                    logging.error(filename)
                    logging.error(img.shape, img.dtype)
                    logging.error(e)
                input_queue.task_done()
                output_queue.put((idx, img))

        # Write workers write resized images to the hdf5 file
        def write_worker():
            num_written = 0
            while True:
                idx, img = output_queue.get()
                if img.ndim == 3:
                    # RGB image, transpose from H x W x C to C x H x W
                    # DO NOT TRANSPOSE
                    imgs_dset[idx] = img
                elif img.ndim == 2:
                    # Grayscale image;
                    # it is H x W so broadcasting to C x H x W will just copy
                    # grayscale values into all channels.

                    img_dtype = img.dtype
                    imgs_dset[idx] = (
                        img[:, :, None] * numpy.array([1, 1, 1])
                    ).astype(img_dtype)
                output_queue.task_done()
                num_written = num_written + 1
                if num_written % 100 == 0:
                    logger.info(
                        'Copied %d / %d images' % (num_written, num_images)
                    )

        # Start the read workers.
        for i in range(num_threads):
            t = threading.Thread(
                target=read_worker,
                args=(img_height, img_width, cls.max_resize))
            t.daemon = True
            t.start()

        # h5py locks internally, so we can only use a single write worker =(
        t = threading.Thread(target=write_worker)
        t.daemon = True
        t.start()

        for idx, filename in enumerate(img_list):
            input_queue.put((idx, filename))

        input_queue.join()
        output_queue.join()

    @classmethod
    def _get_image_filenames(cls, input_dir, num_images):
        """Get a list of image filenames from a directory."""
        img_list = []
        for filename in os.listdir(input_dir):
            _, ext = os.path.splitext(filename)
            if ext in cls.allowed_formats:
                img_list.append(os.path.join(input_dir, filename))
                if num_images and len(img_list) > num_images:
                    break
        return img_list


def download_coco_data(directory):
    """Download and extract the COCO image training data set.

    This file is very large (~13GB) so we check with the user to make
    sure that is ok.

    Args:
        dir - a directory to save the dataset to
    """
    # This is a really big file so ask the user if they are sure they want
    # to start the download.
    if not os.path.isdir(directory):
        logger.info('Creating directory: %s' % directory)
        os.makedirs(directory)

    answer = None
    while answer not in {'Y', 'n'}:
        answer = raw_input(
            'Are you sure you want to download the COCO dataset? [Y/n] '
        )

    if answer == 'n':
        sys.exit()

    logger.info('Downloading COCO image data set. This may take a while...')
    zip_save_path = os.path.join(directory, 'train2014.zip')
    urllib.urlretrieve(_COCO_ZIP_URL, zip_save_path)

    # Files are even bigger to unzip so ask again if they are fine to proceed.
    answer = None
    while answer not in {'Y', 'n'}:
        answer = raw_input(
            'Are you sure you want to unzip things? [Y/n] '
        )

    if answer == 'n':
        sys.exit()

    logger.info('Unzipping COCO image data set. This may take a while...')
    unzip = zipfile.ZipFile(zip_save_path, 'r')
    unzip.extractall(directory)
    unzip.close()
    # Delete the original zipfile
    os.remove(zip_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('Create a dataset to use when training the Fritz'
                     ' Style Transfer model.'))
    parser.add_argument(
        '--output', type=str, required=True,
        help='The name of the resulting h5 dataset.')
    parser.add_argument(
        '--coco-image-dir', type=str, required=True,
        help=('A directory containing a `train2014/` folder with raw extracted'
              'images from the COCO training data set.')
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
        '--download', action='store_true',
        help=('When present, download and extract the COCO image dataset.'
              'Note this is a huge download (~13GB).')
    )
    parser.add_argument(
        '--threads', default=1, type=int,
        help='the number of threads to use to process images.'
    )
    parser.add_argument(
        '--num-images', type=int, help='The number of images to process.'
    )

    args = parser.parse_args()

    if args.download:
        download_coco_data(args.coco_image_dir)

    image_directory = os.path.join(args.coco_image_dir, 'train2014/')
    if not os.path.isdir(image_directory):
        sys.exit(
            'Error: There is no `train2014/` folder in the provided directory.'
        )

    CocoPreprocessor.process_images(
        image_directory,
        args.output,
        image_size=(args.img_height, args.img_width),
        num_images=args.num_images,
        num_threads=args.threads
    )
