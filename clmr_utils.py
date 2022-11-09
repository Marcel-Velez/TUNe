from collections import OrderedDict
import torch
import torch.nn as nn

import os
import yaml

#### from https://github.com/Spijkervet/CLMR/blob/master/clmr/utils/yaml_config_hook.py
def yaml_config_hook(config_file):
    """
    Custom YAML config loader, which can include other yaml files (I like using config files
    insteaad of using argparser)
    """

    # load yaml files in the nested 'defaults' section, which include defaults for experiments
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
        for d in cfg.get("defaults", []):
            config_dir, cf = d.popitem()
            cf = os.path.join(os.path.dirname(config_file), config_dir, cf + ".yaml")
            with open(cf) as f:
                l = yaml.safe_load(f)
                cfg.update(l)

    if "defaults" in cfg.keys():
        del cfg["defaults"]

    return cfg


#### from https://github.com/Spijkervet/CLMR/blob/master/clmr/utils/checkpoint.py
def load_encoder_checkpoint(checkpoint_path: str, output_dim: int, representation_dim: int = 512) -> OrderedDict:
    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    if "pytorch-lightning_version" in state_dict.keys():
        new_state_dict = OrderedDict(
            {
                k.replace("model.encoder.", ""): v
                for k, v in state_dict["state_dict"].items()
                if "model.encoder." in k
            }
        )
    else:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if "encoder." in k:
                new_state_dict[k.replace("encoder.", "")] = v

    new_state_dict["fc.weight"] = torch.zeros(output_dim, representation_dim) # added representation dim argument to Janne Spijkervets code
    new_state_dict["fc.bias"] = torch.zeros(output_dim)
    return new_state_dict
