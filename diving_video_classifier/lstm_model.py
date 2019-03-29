import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import TimeDistributed


def build_lstm_model(input_shape, num_classes, units, input_tensor=None):
    if input_tensor is not None:
        inputs = Input(tensor=input_tensor, shape=input_shape)
    else:
        inputs = Input(shape=input_shape)

    out = LSTM(units, dropout=0.3)(inputs)
    out = Dense(units * 2, activation='relu')(out)
    out = Dropout(0.4)(out)
    out = Dense(num_classes, activation='softmax')(out)

    return Model(inputs=inputs, outputs=out)


def build_training_model(batch_size,
                         image_size,
                         sequence_length,
                         classes,
                         encoded_diving_tfrecord,
                         units=256,
                         lr=0.001):
    dataset = encoded_diving_tfrecord.build_tf_dataset(
        image_size, batch_size, sequence_length
    )

    iterator = dataset.make_one_shot_iterator()
    example = iterator.get_next()
    model = build_lstm_model(
        (sequence_length, 1280),
        len(classes),
        units,
        input_tensor=example['image']
    )
    print(model.summary())

    output = tf.one_hot(example['label'], depth=3)
    output = output[:, sequence_length - 1, :]

    optimizer = keras.optimizers.Adam(lr=lr)
    model.compile(
        optimizer,
        loss=keras.losses.categorical_crossentropy,
        metrics=['categorical_accuracy'],
        target_tensors=output
    )
    # model.load_weights('trained_lstm.h5')

    return model


def build_lstm_model_multiple_outputs(
        input_shape, num_classes, units, input_tensor=None):
    if input_tensor is not None:
        inputs = Input(tensor=input_tensor, shape=input_shape)
    else:
        inputs = Input(shape=input_shape)

    out = LSTM(units, return_sequences=True, dropout=0.3)(inputs)
    out = Dense(units * 2, activation='relu')(out)
    out = Dropout(0.4)(out)
    out = TimeDistributed(Dense(num_classes, activation='softmax'))(out)

    return Model(inputs=inputs, outputs=out)


def build_training_model_multiple(batch_size,
                                  image_size,
                                  sequence_length,
                                  classes,
                                  encoded_diving_tfrecord,
                                  units=256,
                                  lr=0.001):
    dataset = encoded_diving_tfrecord.build_tf_dataset(
        image_size, batch_size, sequence_length
    )

    iterator = dataset.make_one_shot_iterator()
    example = iterator.get_next()
    model = build_lstm_model_multiple_outputs(
        (sequence_length, 1280),
        len(classes),
        units,
        input_tensor=example['image']
    )
    print(model.summary())

    output = tf.one_hot(example['label'], depth=3)

    optimizer = keras.optimizers.Adam(lr=lr)
    model.compile(
        optimizer,
        loss=keras.losses.categorical_crossentropy,
        metrics=['categorical_accuracy'],
        target_tensors=output
    )
    # model.load_weights('trained_lstm.h5')

    return model
