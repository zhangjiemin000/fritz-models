
download:
	./download_and_convert_ade20k.sh

create-training-data:
	mkdir -p data/${LABEL_SET}
	python create_tfrecord_dataset.py \
		-i data/ADEChallengeData2016/images/training/ \
		-a data/ADEChallengeData2016/annotations/training/ \
		-o data/${LABEL_SET}/${LABEL_SET}_data.tfrecord \
		-l data/ADEChallengeData2016/objectInfo150.txt \
		-w "person, individual, someone, somebody, mortal, soul|house:building, edifice:house:skyscraper|sky|car, auto, automobile, machine, motorcar:bus, autobus, coach, charabanc, double-decker, jitney, motorbus, motorcoach, omnibus, passenger vehicle:truck, motortruck:van|bicycle, bike, wheel, cycle:minibike, motorbike" \
		-t 0.20

upload-data:
	gsutil cp data/${LABEL_SET}/* gs://${GCS_BUCKET}/data/${LABEL_SET}/


train-local:
	pipenv run python -m image_segmentation.train \
	    --data gs://fritz-data-sandbox/ADEChallengeData2016/people/people_data.tfrecord \
	    -l gs://fritz-data-sandbox/ADEChallengeData2016/people/labels.txt \
	    -n 20 \
	    -s 384 \
	    -a 1 \
	    --batch-size 1 \
	    --steps-per-epoch 10 \
	    --parallel-calls 4 \
	    --lr 0.001 \
	    --model-name people_local_test


train-cloud:
	python setup.py sdist
	gcloud ml-engine jobs submit training `whoami`_image_segmentation_`date +%s` \
	    --runtime-version 1.9 \
	    --job-dir=gs://${GCS_BUCKET} \
	    --packages dist/image_segmentation-1.0.tar.gz,nvidia_dali-0.4.1-38228-cp27-cp27mu-manylinux1_x86_64.whl \
	    --module-name image_segmentation.train \
	    --region us-central1 \
	    --config config.yaml \
	    -- \
	    -d gs://fritz-data-sandbox/ADEChallengeData2016/people/people_data.tfrecord \
	    -l gs://fritz-data-sandbox/ADEChallengeData2016/people/labels.txt \
	    --use-dali \
	    -n 5000 \
	    -s 768 \
	    -a 1 \
	    --batch-size 12 \
	    --steps-per-epoch 250 \
	    --parallel-calls 4 \
	    --lr 0.001 \
	    --add-noise \
	    --model-name ${MODEL_NAME} \
	    --gcs-bucket gs://${GCS_BUCKET}/train
