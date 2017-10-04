# fritz-style-transfer
Code for training artistic style transfer models with Keras and converting them to Core ML.

# Preprocessing Training Data
The training data comes from the [COCO Training data set](http://cocodataset.org/). It consists of ~80,000 images and labels, although the labels arent used here.

the `create_training_dataset.py` script will download and unzip this data then process images to create an h5 dataset used by the style transfer network trainer. You can run this with the command below. Note the first time you run this you will need to download and unzip 13GB worth of data and it can take a while. The command only processes the first 10 images to make sure things are working, but you can modify `--num-images` to process more.

```
python create_training_dataset.py \
--output data/training_images.h5 \
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
--training-image-dset data/training_images.h5 \
--style-images data/starry-night.jpg \
--weights-checkpoint data/starry_night_keras_weights.h5 \
--img-height 256 \
--img-width 256 \
--log-interval 1 \
--num-iterations 10
```

If everything looks good, we can pick up where we left off and keep training the same model.

```
python train_network.py \
--training-image-dset data/training_images.h5 \
--style-images data/starry-night.jpg \
--weights-checkpoint data/starry_night_keras_weights.h5 \
--img-height 256 \
--img-width 256 \
--num-iterations 100 \
--fine-tune
```

# Stylizing Images
To stylize an image with a trained model you can run:

```
python stylize_image.py \
--input-image data/test.jpg \
--output-image data/stylized_test.jpg \
--weights-checkpoint data/starry_night_keras_weights.h5
```

# TODO
* Finish added conversion tools for Core ML
* Clean up structure