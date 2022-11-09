for DATASET_TAG in  gtzan giantsteps emomusic magnatagatune
do
	python -m jukemir.datasets.cache $DATASET_TAG audio
done
