import cv2
import PIL.Image
from diving_video_classifier import build_tfrecord_classifications
from utils import tfrecord_helpers
from diving_video_classifier import diving_footage
from diving_video_classifier import encoded_diving_footage
from diving_video_classifier import lstm_model
import logging
import numpy
import skimage.filters
logger = logging.getLogger(__name__)


def classify_video(filename, frame_classifications):
    video = cv2.VideoCapture(filename)
    total = 0

    while video.isOpened():
        # Capture frame-by-frame
        ret, frame = video.read()
        if not ret:
            break
        frame_id = int(video.get(1))

        classification = frame_classifications[frame_id]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield PIL.Image.fromarray(frame), frame_id, classification

        total += 1

    # When everything done, release the video capture object
    video.release()


def generate_frame_list(frame_splits, categories):
    """Build categories that frames belong to based on frame splits."""
    if len(frame_splits) != len(categories):
        raise Exception(f"Need to pass list of {categories}")

    frames = []
    split = 0
    for i in range(len(categories)):
        next_split = frame_splits[i]
        frames.extend(categories[i] for _ in range(split, next_split))
        split = next_split

    return frames


def build_dataset(video_config, categories):
    """Building and saving original videos into one tfrecord file. """
    for filename, splits in video_config:
        logger.info(f'beginning file {filename}')
        classifications = generate_frame_list(splits, categories)

        classifications = classify_video(filename, classifications)
        for image, frame_id, classification in classifications:
            yield diving_footage.DivingTFRecord.build_example(
                image, filename, frame_id, classification)


def prepare_image_for_mobilenet_prediction(image):
    image = image.resize((224, 224))
    image_data = numpy.array(image)
    image_data = image_data * 1. / 128. - 1.0
    return skimage.filters.gaussian(image_data, sigma=0.0)


def generate_encoded_tfrecord(records,
                              encoder_model,
                              sequence_size,
                              categories):
    reverse_categories = {v: k for k, v in categories.items()}
    results = []
    classes = []
    current_filename = ''
    for i, record in enumerate(records):
        example = diving_footage.DivingTFRecord.decode_single_example(record)
        if example['filename'] != current_filename:
            current_filename = example['filename']
            classes = []
            results = []

        image = prepare_image_for_mobilenet_prediction(example['image'])
        result = encoder_model.predict(image[None, :, :, :])
        classification = example['classification'].decode('utf-8')
        category = reverse_categories[classification]

        if len(results) < sequence_size:
            results.append(result)
            classes.append(category)
            continue

        combined = numpy.concatenate(results)
        yield encoded_diving_footage.EncodedDivingTFRecord.build_example(
            combined, classes
        )
        results = results[:-1] + [result]
        classes = classes[:-1] + [category]


def evaluate_trained_model(encoder_model,
                           trained_model,
                           video_records,
                           sequence_length,
                           categories,
                           lstm_units):

    model = lstm_model.build_lstm_model(
        (sequence_length, 1280),
        len(categories),
        lstm_units)
    model.set_weights(trained_model.get_weights())

    results = []
    images = []
    prepare_image = prepare_image_for_mobilenet_prediction
    for i, record in enumerate(video_records):
        example = video_records.decode_single_example(record)

        image = prepare_image(example['image'])
        result = encoder_model.predict(image[None, :, :, :])
        # TODO: add filename check
        if len(results) < sequence_length:
            results.append(result)
            images.append(example['image'])
            continue

        combined = numpy.concatenate(results)
        predictions = model.predict(combined[None, :, :])
        yield i, example['image'], predictions, example['classification']

        results = []
        images = []


def evaluate_trained_model_multiple(encoder_model,
                                    trained_model,
                                    video_records,
                                    sequence_length,
                                    categories,
                                    lstm_units):
    model = lstm_model.build_lstm_model_multiple_outputs(
        (sequence_length, 1280),
        len(categories),
        lstm_units)
    model.set_weights(trained_model.get_weights())

    results = []
    images = []
    prepare_image = prepare_image_for_mobilenet_prediction
    for i, record in enumerate(video_records):
        example = video_records.decode_single_example(record)

        image = prepare_image(example['image'])
        result = encoder_model.predict(image[None, :, :, :])
        # TODO: add filename check
        if len(results) < sequence_length:
            results.append(result)
            images.append(example['image'])
            continue

        combined = numpy.concatenate(results)
        predictions = model.predict(combined[None, :, :])
        yield i, example['image'], predictions, example['classification']

        results = []
        images = []
