# Fritz Style Transfer
Artistic style tranfer models transfer the style of an image onto the content of another.

This repository contains code for training mobile-friendly style transfer models.

<img src="https://github.com/fritzlabs/fritz-models/blob/master/style_transfer/example/starry_night_results.jpg" width="662" height="295">

Left: Original image. Middle: Image stylzed with a 17kb small model. Right: Image stylzed by the default large model.

## Add style transfer to your app in minutes with Fritz

If you're looking to add style transfer to your app quickly, check out [Fritz](https://fritz.ai/?utm_source=github&utm_campaign=fritz-models&utm_content=style-transfer). The Fritz SDK provides 11 pre-trained style transfer models along with all the code you need to apply them images or live video. If you want to train your own model, keep reading.

## Train your own custom style model in 20 minutes

You can now train your own personal style transfer model in about 20 minutes using Fritz Style Transfer and Google Colab. Just create your own playground from [this notebook](https://colab.research.google.com/drive/1nDkxLKBgZGFscGoF0tfyPMGqW03xITl0#scrollTo=L9aTwLIqtFTE) to get started. You can read more about how it works [here](https://heartbeat.fritz.ai/20-minute-masterpiece-4b6043fdfff5?utm_source=github&utm_campaign=fritz-models&utm_content=style-transfer).

## Installation

If you're not installing using a package manager like `pip`, make sure the root directory is on your `PYTHONPATH`:

```
export PYTHONPATH=$PYTHONPATH:`pwd`
```

## Preprocessing Training Data
The training data comes from the [COCO Training data set](http://cocodataset.org/). It consists of ~80,000 images and labels, although the labels arent used here.

the `create_training_dataset.py` script will download and unzip this data then process images to create an h5 dataset used by the style transfer network trainer. You can run this with the command below. Note the first time you run this you will need to download and unzip 13GB worth of data and it can take a while. The command only processes the first 10 images to make sure things are working, but you can modify `--num-images` to process more.

```
python create_training_dataset.py \
--output example/training_images.tfrecord \
--image-dir path/to/coco/ \
--num-images 10
```

Note that if you have already downloaded and extracted a set of images to use for training, that directory needs to be called `train2014/` and you need to point `--coco-image-dir` to the parent directory that contains that folder. Otherwise you can use the `--download` flag.

## Training a Style Transfer Model

To train the model from scratch for 100 iterations:

```
python style_transfer/train.py \
--training-image-dset example/training_images.tfrecord \
--style-images example/starry_night.jpg \
--model-checkpoint example/starry_night.h5 \
--image-size 256,256 \
--alpha 0.25 \
--log-interval 1 \
--num-iterations 10
```

If everything looks good, we can pick up where we left off and keep training the same model.

```
python style_transfer/train.py \
--training-image-dset example/training_images.tfrecord \
--style-images example/starry_night.jpg \
--model-checkpoint example/starry_night.h5 \
--image-size 256,256 \
--alpha 0.25 \
--num-iterations 1000 \
--fine-tune-checkpoint example/starry_night.h5
```

If you're using the full COCO dataset, you'll need around 20,000 iterations to train a model from scratch with a batch size of 24. If you're starting from a pre-trained model checkpoint, 5,000 steps should work. A model pre-trained on Starry Night is provided in the `example/` folder.

For styles that are abstract with strong geometric patters, try higher values for `--content-weight` like `3` or `10`. For styles that are more photo-realistic images with smaller details, boost the `--style-weight` to `0.001` or more.

Finally, note that for training, we resize images to be 256x256px. This is for training only. Final models can be set to take images of any size.

### Training models for mobile

By default, the style transfer networks produced here are roughly 7mb in size and contain 7 million parameters. They can create a stylized image in ~500ms on high end mobile phones, and 5s on lower end phones. To make the model's faster, we've included a width-multiplier parameter similar to the one introduced by Google in their MobileNet architecture. The value `alpha` can be set between 0 and 1 and will control how many filters are included in each layer. Lower `alpha` means fewer filters, fewer parameters, faster models, with slightly worse style transfer abilities. In testing, `alpha=0.25` produced models that ran at 17fps on an iPhone X, while still transfering styles well.

Finally, for models that are intended to be used in real-time on a CPU only, you can use the `--use-small-network` flag to train a model architecture that has been heavily pruned. The style transfer itself isn't quite as good, but the results are usable and the models are incredible small.

## Stylizing Images
To stylize an image with a trained model you can run:

```
python stylize_image.py \
--input-image example/dog.jpg \
--output-image example/stylized_dog.jpg \
--model-checkpoint example/starry_night_256x256_025.h5
```

## Convert to Mobile
Style transfer models can be converted to both Core ML and TensorFlow Mobile formats.

### Convert to Core ML
Use the converter script to convert to Core ML.

This converter is a slight modification of Apple's keras converter that allows
the user to define custom conversions between Keras layers and core ml layers. This allows us to convert the Instance Normalization and Deprocessing layers.

```
python convert_to_coreml.py \
--keras-checkpoint example/starry_night_256x256_025.h5 \
--alpha 0.25 \
--image-size 640,480 \
--coreml-model example/starry_night_640x480_025.mlmodel
```

### Convert to TensorFlow Mobile
Models cannot be converted to TFLite because some operations are not supported, but TensorFlow Mobile works fine. To convert your model to an optimized frozen graph, run:

```
python convert_to_tfmobile.py \
--keras-checkpoint example/starry_night_256x256_025.h5 \
--alpha 0.25 \
--image-size 640,480 \
--output-dir example/
```

This produces a number of TensorFlow graph formats. The `*_optimized.pb` graph file is the one you want to use with your app. Note that the input node name is `input_1` and the output node name is `deprocess_stylized_image_1/mul`.

## Train on Google Cloud ML

This library is designed to work with certain configurations on Google Cloud ML so you can train styles in parallel and take advantage GPUs. Assuming you have Google Cloud ML and Google Cloud Storage set up, the following commands will get you training new models in just a few hours.

### Set up your Google Cloud Storage bucket.

This repo assumes the structure on Google Cloud is 

```
gs://${YOUR_GCS_BUCKET}/
    |-- data/
        |-- training_images.tfrecord
        |-- starry_night_256x256_025.h5
        |-- style_images/
            |-- style_1.jpg
            |-- style_2.jpg
    |-- dist/
        |-- fritz_style_transfer.zip
    |-- train/
        |-- pretrained_model.h5
        |-- output_model.h5
```

To make things easier, start by setting some environmental variables.

```
export YOUR_GCS_BUCKET=your_gcs_bucket
export FRITZ_STYLE_TRANSFER_PATH=/path/to/fritz-models/style_transfer/
export KERAS_CONTRIB_PATH=/path/to/keras-contrib
export STYLE_NAME=style_name
```

Note that `STYLE_NAME` should be the filename of the style image (without the extension).

Create the GCS bucket if you haven't already:

```
gsutil mb gs://${YOUR_GCS_BUCKET}
```

Copy training data to GCS, pre-trained checkpoints, and style image to:
```
gsutil cp example/training_images.tfrecord gs://${YOUR_GCS_BUCKET}/data
gsutil cp example/${STYLE_NAME}.jpg gs://${YOUR_GCS_BUCKET}/data/style_images/
gsutil cp example/starry_night_256x256_025.h5 gs://${YOUR_GCS_BUCKET}/data/
```

### Package up libraries.

Zip up all of the local files to send up to Google Cloud.
```
python setup.py sdist
```

Zip up keras_contrib so it's available to the library as well.
```
pushd ${KERAS_CONTRIB_PATH}
python setup.py sdist
cp dist/* ${FRITZ_STYLE_TRANSFER_PATH}/dist/
popd
```

### Start the training job

The following command will start training a new style transfer models from a pre-trained checkpoint. This configuration trains on 256x256 images and has `--alpha=0.25` making it suitable for real-time use in mobile apps.

```
gcloud ml-engine jobs submit training `whoami`_style_transfer`date +%s` \
    --runtime-version 1.8 \
    --job-dir=gs://${YOUR_GCS_BUCKET} \
    --packages dist/style_transfer-1.0.tar.gz,dist/keras_contrib-2.0.8.tar.gz \
    --module-name style_transfer.train \
    --region us-east1 \
    --scale-tier basic_gpu \
    -- \
    --training-image-dset gs://${YOUR_GCS_BUCKET}/data/test_training_images.tfrecord \
    --style-images gs://${YOUR_GCS_BUCKET}/data/style_images/${STYLE_NAME}.jpg \
    --model-checkpoint ${STYLE_NAME}_256x256_025.h5 \
    --image-size 256,256 \
    --alpha 0.25 \
    --num-iterations 5000 \
    --batch-size 24 \
    --content-weight 1 \
    --style-weight .0001 \
    --gcs-bucket gs://${YOUR_GCS_BUCKET}/train \
    --fine-tune-checkpoint gs://${YOUR_GCS_BUCKET}/data/starry_night_256x256_025.h5
```

Distributed training and TPUs are not yet supported.

## Add the model to your app with Fritz

Now that you have a style transfer model that works for both iOS and Android, head over to [https://fritz.ai](https://fritz.ai/?utm_source=github&utm_campaign=fritz-models&utm_content=style-transfer) for tools to help you integrate it into your app and manage it over time.

## What's next?

* Get a free [Fritz account](https://www.fritz.ai?utm_source=github&utm_campaign=fritz-models&utm_content=style-transfer).
* Read the [docs](https://docs.fritz.ai?utm_source=github&utm_campaign=fritz-models&utm_content=style-transfer).
* Learn more about mobile machine learning on [Heartbeat](https://heartbeat.fritz.ai/?utm_source=github&utm_campaign=fritz-models&utm_content=style-transfer).
* Stay up-to-date with the [Heartbeat Newsletter](http://eepurl.com/c_verH)
* Join us [on Slack](https://join.slack.com/t/heartbeat-by-fritz/shared_invite/enQtNTI4MDcxMzI1MzAwLWIyMjRmMGYxYjUwZmE3MzA0MWQ0NDk0YjA2NzE3M2FjM2Y5MjQxMWM2MmQ4ZTdjNjViYjM3NDE0OWQxOTBmZWI).
* Follow us on Twitter: [@fritzlabs](https://twitter.com/fritzlabs)