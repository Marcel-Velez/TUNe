# majority of the code is copy pasted from the torchaudio GTZAN class (https://pytorch.org/audio/stable/_modules/torchaudio/datasets/gtzan.html), 
# slightly altered because the download link to the dataset was unavailable at the time of writing.

import torchaudio
from torchaudio.datasets.gtzan import gtzan_genres, filtered_test, filtered_train, filtered_valid
from clmr.datasets import Dataset
import os 
from typing import Any, Tuple, Optional, Union
from torch import Tensor
from pathlib import Path
from torch.hub import download_url_to_file

# # https://web.archive.org/web/20220328223413/http://opihi.cs.uvic.ca/sound/genres.tar.gz

from torchaudio.datasets.utils import (
    download_url,
    extract_archive,
)



FOLDER_IN_ARCHIVE = "gtzan"


def load_gtzan_item(fileid: str, path: str, ext_audio: str) -> Tuple[Tensor, str]:
    """
    Loads a file from the dataset and returns the raw waveform
    as a Torch Tensor, its sample rate as an integer, and its
    genre as a string.
    """
    # Filenames are of the form label.id, e.g. blues.00078
    label, _ = fileid.split(".")

    # Read wav
    file_audio = os.path.join(path, label, fileid + ext_audio)
    waveform, sample_rate = torchaudio.load(file_audio)

    return waveform, sample_rate, label



URL = "https://web.archive.org/web/20220328223413/http://opihi.cs.uvic.ca/sound/genres.tar.gz"#http://opihi.cs.uvic.ca/sound/genres.tar.gz"
FOLDER_IN_ARCHIVE = "genres"
_CHECKSUMS = {
    "http://opihi.cs.uvic.ca/sound/genres.tar.gz": "24347e0223d2ba798e0a558c4c172d9d4a19c00bb7963fe055d183dadb4ef2c6"
}




class GTZAN(Dataset):
    """Create a Dataset for GTZAN.

    Note:
        Please see http://marsyas.info/downloads/datasets.html if you are planning to use
        this dataset to publish results.

    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        url (str, optional): The URL to download the dataset from.
            (default: ``"http://opihi.cs.uvic.ca/sound/genres.tar.gz"``)
        folder_in_archive (str, optional): The top-level directory of the dataset.
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
        subset (str or None, optional): Which subset of the dataset to use.
            One of ``"training"``, ``"validation"``, ``"testing"`` or ``None``.
            If ``None``, the entire dataset is used. (default: ``None``).
    """

    _ext_audio = ".wav"

    def __init__(
        self,
        root: Union[str, Path],
        url: str = URL,
        folder_in_archive: str = FOLDER_IN_ARCHIVE,
        download: bool = False,
        subset: Optional[str] = None,
    ) -> None:

        # super(GTZAN, self).__init__()

        # Get string representation of 'root' in case Path object is passed
        root = os.fspath(root)


        self.root = root
        self.url = url
        self.folder_in_archive = folder_in_archive
        self.download = download
        # self.subset = subset

        if subset:
            subset = {"train":"training", "valid":"validation", "test":"testing"}[subset]
        self.subset = subset

        assert subset is None or subset in ["training", "validation", "testing"], (
            "When `subset` not None, it must take a value from " + "{'training', 'validation', 'testing'}."
        )

        archive = os.path.basename(url)
        archive = os.path.join(root, archive)
        self._path = os.path.join(root, folder_in_archive)

        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    checksum = _CHECKSUMS.get(url, None)
                    download_url_to_file(url, archive, hash_prefix=checksum)
                extract_archive(archive)

        if not os.path.isdir(self._path):
            raise RuntimeError("Dataset not found. Please use `download=True` to download it.")

        if self.subset == "training":
            self._walker = filtered_train
        elif self.subset == "validation":
            self._walker = filtered_valid
        elif self.subset == "testing":
            self._walker = filtered_test
        else:
            self._walker = filtered_test+filtered_train+filtered_valid

        self.labels = gtzan_genres
        self.label2idx = {}
        for idx, label in enumerate(self.labels):
            self.label2idx[label] = idx

        # self.fl, self.binary = get_file_list(self._path, self.subset, self.split)
        self.n_classes = len(self.label2idx.keys())


    def __getitem__(self, n: int) -> Tuple[Tensor, int, str]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            (Tensor, int, str): ``(waveform, sample_rate, label)``
        """
        fileid = self._walker[n]
        item = load_gtzan_item(fileid, self._path, self._ext_audio)
        waveform, _, label = item
        return waveform, self.label2idx[label]


    def __len__(self) -> int:
        return len(self._walker)


    def file_path(self, n: int) -> str:
        fileid = self._walker[n]
        label, _ = fileid.split(".")
        # Read wav
        file_audio = os.path.join(self._path, label, fileid + self._ext_audio)
        return file_audio

    # def __getitem__(self, idx):
    #     audio, sr, label = self.dataset[idx]
    #     print(audio)
    #     label = self.label2idx[label]
    #     return audio, label

    # def __len__(self):
    #     return len(self.dataset)
