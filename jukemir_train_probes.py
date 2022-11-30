import json
import pathlib

from jukemir import CACHE_PROBES_DIR
from jukemir.probe import paper_grid_cfgs, ProbeExperimentConfig, execute_probe_experiment
from tqdm import tqdm

DATASETS = ["gtzan_ff", "giantsteps_clips", "magnatagatune", "emomusic"]
REPRESENTATIONS = ["TunePlus"]  # ["Tune5Tail"]

print(f"creating all executable probe settings for datasets: {DATASETS} and representations: {REPRESENTATIONS}")
for dataset in DATASETS:
    for representation in REPRESENTATIONS:
        grid_dir = pathlib.Path(CACHE_PROBES_DIR, dataset, representation)
        grid_dir.mkdir(parents=True, exist_ok=True)
        for cfg in paper_grid_cfgs(dataset, representation):
            with open(pathlib.Path(grid_dir, f"{cfg.uid()}.json"), "w") as f:
                f.write(json.dumps(cfg, indent=2, sort_keys=True))

print(f"done creating executable settings")
# from jukemir reproduce/5_grid_train_serial.py

for train_dataset in DATASETS:
    for train_representation in REPRESENTATIONS:
        grid_dir = pathlib.Path(CACHE_PROBES_DIR, train_dataset, train_representation)
        grid_cfgs = sorted(list(grid_dir.glob("*.json")))
        print(f"Training {len(grid_cfgs)} probes for {dataset} {representation}")
        for cfg_path in tqdm(list(grid_dir.glob("*.json"))):
            with open(cfg_path, "r") as f:
                cfg = ProbeExperimentConfig(json.load(f))
            execute_probe_experiment(cfg, wandb=False)