
num_workers=$(nproc --all)
echo "Num threads: $num_workers"


#  magnatagatune 
for ASSET_PREFIX in gtzan giantsteps emomusic magnatagatune
do
	python -m jukemir.assets $ASSET_PREFIX --delete_wrong --num_parallel $num_workers
done

