
num_workers=$(sysctl -n hw.ncpu)
echo "Num threads: $num_workers"


for ASSET_PREFIX in  giantsteps emomusic magnatagatune gtzan
do
	python3 -m jukemir.assets $ASSET_PREFIX --delete_wrong --num_parallel $num_workers
done

for DATASET_TAG in  giantsteps_clips emomusic magnatagatune gtzan_ff
do
	python3 -m jukemir.datasets.cache $DATASET_TAG audio
done