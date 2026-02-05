import soundfile as sf
import torch
from utils import handle_scp


def read_wav(fname, return_rate=False):
    """
    Read wav file using soundfile (Windows-safe)
    Returns tensor [C, L]
    """
    data, sr = sf.read(fname, dtype="float32")

    if data.ndim == 1:
        data = data[None, :]
    else:
        data = data.T

    src = torch.from_numpy(data)

    if return_rate:
        return src, sr
    return src


def write_wav(fname, src, sample_rate):
    src = src.detach().cpu().numpy()
    if src.ndim == 2:
        src = src.T
    sf.write(fname, src, sample_rate)


class AudioReader(object):
    def __init__(self, scp_path, sample_rate=8000):
        self.sample_rate = sample_rate
        self.index_dict = handle_scp(scp_path)
        self.keys = list(self.index_dict.keys())

    def _load(self, key):
        src, sr = read_wav(self.index_dict[key], return_rate=True)
        if sr != self.sample_rate:
            raise RuntimeError(f"Sample rate mismatch {sr} != {self.sample_rate}")
        return src

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        if isinstance(index, int):
            key = self.keys[index]
        else:
            key = index
        return self._load(key)
