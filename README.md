# fritz-style-transfer
Code for training artistic style transfer models with Keras and converting them to Core ML.

# Preprocessing Training Data
The training data comes from the [COCO Training data set](http://cocodataset.org/). It consists of ~80,000 images and labels, although the labels arent used here.

the `create_training_dataset.py` script will download and unzip this data then process images to create an h5 dataset used by the style transfer network trainer. You can run this with the command below. Note the first time you run this you will need to download and unzip 13GB worth of data and it can take a while. The command only processes the first 10 images to make sure things are working, but you can modify `--num-images` to process more.

```
python create_training_dataset.py \
--output data/training_images.tfrecord \
--coco-image-dir data/ \
--img-height 256 \
--img-width 256 \
--threads 1 \
--num-images 10
```

Note that if you have already downloaded and extracted a set of images to use for training, that directory needs to be called `train2014/` and you need to point `--coco-image-dir` to the parent directory that contains that folder.

# Training a Style Transfer Model
To train the model from scratch for 100 iterations:

```
python train_network.py \
--training-image-dset data/training_images.tfrecord \
--style-images data/starry-night.jpg \
--model-checkpoint data/starry_night_keras.h5 \
--image-size 256,256 \
--log-interval 1 \
--num-iterations 10
```

If everything looks good, we can pick up where we left off and keep training the same model.

```
python train_network.py \
--training-image-dset data/training_images.tfrecord \
--style-images data/starry-night.jpg \
--model-checkpoint data/starry_night_keras.h5 \
--image-size 256,256 \
--num-iterations 100 \
--fine-tune-checkpoint data/starry_night_keras.h5
```

# Stylizing Images
To stylize an image with a trained model you can run:

```
python stylize_image.py \
--input-image data/test.jpg \
--output-image data/stylized_test.jpg \
--model-checkpoint data/starry_night_keras.h5
```


# Convert to Core ML
Use the converter script to convert to Core ML.

This converter is a slight modification of Apple's keras converter that allows
the user to define custom conversions between Keras layers and core ml layers. This allows us to convert the Instance Normalization and Deprocessing layers.

```
python convert_to_coreml.py \
--keras-checkpoint data/starry_night_keras.h5 \
--coreml-model data/starry_night.mlmodel
```

# Train on Google Cloud ML

Zip up all of the local files to send up to Google Cloud
```
python setup.py sdist
```

# Zip up keras_contrib so it's available
```
pushd path/to/keras-contrib
python setup.py sdist
cp dist/* ~/fritz/fritz-style-transfer/dist/
popd
```

```
# from `fritz-style-transfer/`
export STYLE_NAME=name_of_style
gcloud ml-engine jobs submit training `whoami`_style_transfer`date +%s` \
    --runtime-version 1.8 \
    --job-dir=gs://${YOUR_GCS_BUCKET} \
    --packages dist/style_transfer-1.0.tar.gz,dist/keras_contrib-2.0.8.tar.gz \
    --module-name style_transfer.train \
    --region us-east1 \
    --scale-tier basic_gpu \
    -- \
    --training-image-dset gs://fritz-data-sandbox/mscoco/tfrecord/coco_train.record \
    --style-images gs://${YOUR_GCS_BUCKET}/style_images/${STYLE_NAME}.jpg \
    --model-checkpoint ${STYLE_NAME}_256x256_025.h5 \
    --image-size 256,256 \
    --alpha 0.25 \
    --num-iterations 5000 \
    --batch-size 24 \
    --content-weight 1 \
    --style-weight .001 \
    --gcs-bucket gs://${YOUR_GCS_BUCKET}/train \
    --fine-tune-checkpoint gs://${YOUR_GCS_BUCKET}/train/starry_night_256x256_025.h5
```
