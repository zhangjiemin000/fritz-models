import os
import glob
import pathlib
import PIL.Image
from matplotlib import pyplot
from collections import defaultdict
from utils import tfrecord_helpers
import numpy
BASE = '/Users/chris/Downloads/DAVIS'

CATEGORIES = [
    'bike-packing',
    'boxing-fisheye',
    'breakdance-flare',
    'cat-girl',
    'crossing',
    'disc-jockey',
    'drone',
    'hike',
    'hockey',
    'horsejump-high',
    'horsejump-low',
    'kid-football',
    'kite-surf',
    'kite-walk',
    'lab-coat',
    'lindy-hop',
    'loading',
    'longboard',
    'lucia',
    'motocross-bumps',
    'motocross-jump',
    'parkour',
    'rollerblade',
    'schoolgirls',
    'scooter-gray',
    'shooting',
    'snowboard',
    'stroller',
    'stunt',
    'swing',
    'tennis',
    'tuk-tuk',
    'upside-down',
    'walking',
]

people_indices_by_category = {
    'bike-packing': [2],
    'boxing-fisheye': [1, 2, 3],
    'breakdance-flare': [1],
    'cat-girl': [1],
    'crossing': [1, 2],
    'disc-jockey': [1, 2, 3],
    'drone': [3, 5],
    'hike': [1],
    'hockey': [1],
    'horsejump-high': [2],
    'horsejump-low': [2],
    'kid-football': [1],
    'kite-surf': [3],
    'kite-walk': [2],
    'lab-coat': [3, 4, 5],
    'lindy-hop': [1, 2, 3, 4, 5, 6, 7, 8],
    'loading': [1, 3],
    'longboard': [4, 5],
    'lucia': [1],
    'motocross-bumps': [1],
    'motocross-jump': [1],
    'parkour': [1],
    'rollerblade': [1],
    'schoolgirls': [1, 3, 5, 7],
    'scooter-gray': [2],
    'shooting': [2],
    'snowboard': [2],
    'stroller': [1],
    'stunt': [2],
    'swing': [1],
    'tennis': [1],
    'tuk-tuk': [1, 2, 3],
    'upside-down': [1, 2],
    'walking': [1, 2]
}


def get_image_folder(category):
    return os.path.join(BASE, 'JPEGImages/480p', category)


def get_annotation_folder(category):
    return os.path.join(BASE, 'Annotations/480p', category)


def load_images_for_category(category):
    image_folder = get_image_folder(category)
    annotation_folder = get_annotation_folder(category)

    for path in glob.glob(os.path.join(image_folder, '*')):
        path = pathlib.Path(path)
        annotation_path = os.path.join(annotation_folder, path.stem + '.png')
        annotation_path = pathlib.Path(annotation_path)
        yield PIL.Image.open(path), PIL.Image.open(annotation_path)


def get_values_for_category(category):
    images_by_index = defaultdict(list)

    for image, mask in load_images_for_category(category):

        for i in numpy.unique(mask):
            images_by_index[i].append((image, mask))

    return images_by_index


def pick_indices(category, num_photos_to_iterate=3):
    print(f"Category: {category}")

    images_by_index = get_values_for_category(category)
    for index, images in images_by_index.items():
        print(f"Image with index {index}")
        image, mask = images[0]
        pyplot.imshow(image)
        pyplot.show()
        mask_array = numpy.array(mask)
        new_array = numpy.array(mask_array)
        new_array[new_array != index] = 0
        pyplot.imshow(new_array)
        pyplot.show()


def yield_new_mask_values(category):
    people_indices = numpy.array(people_indices_by_category[category])

    for image, mask in load_images_for_category(category):
        mask_array = numpy.array(mask)
        for i in numpy.unique(mask_array):
            if i in people_indices:
                mask_array[mask_array == i] = 1
            else:
                mask_array[mask_array == i] = 0

        yield image, mask_array


def create_new_tfrecords():
    for category in CATEGORIES:
        print(category)
        for image, mask in yield_new_mask_values(category):
            yield tfrecord_helpers.build_example(f'{category}/file',
                                                 image,
                                                 mask)
