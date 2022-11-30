for DATASET in giantsteps_clips magnatagatune emomusic gtzan_ff
do
	for REPRESENTATION in TunePlus  # Tune5Tail
	do
		docker exec -it jukemir \
			python -m jukemir.probe.aggregate \
				$DATASET \
				$REPRESENTATION \
				--evaluate test \
				--expected_num_runs 216
	done
done
