import os
from .dataset import Dataset
from .audio import AUDIO
from .librispeech import LIBRISPEECH
from .gtzan import GTZAN
from .magnatagatune import MAGNATAGATUNE
from .million_song_dataset import MillionSongDataset


def get_dataset(dataset, dataset_dir, subset, download=True):

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    if dataset == "audio":
        d = AUDIO(root=dataset_dir)
    elif dataset == "fma":
        d = AUDIO(root=os.path.join(dataset_dir,'fma'), src_ext_audio=".mp3")
    elif dataset == "mcgill":
        d = AUDIO(root=os.path.join(dataset_dir,'mcgill'), src_ext_audio=".wav")
    elif dataset == "librispeech":
        d = LIBRISPEECH(root=dataset_dir, download=download, subset=subset)
    elif dataset == "gtzan":
        d = GTZAN(root=os.path.join(dataset_dir,"gtzan"), download=False, subset=subset)
    elif dataset == "magnatagatune":
        d = MAGNATAGATUNE(root=dataset_dir, download=download, subset=subset)
    elif dataset == "msd":
        d = MillionSongDataset(root=dataset_dir, subset=subset)
    else:
        raise NotImplementedError("Dataset not implemented")
    return d
