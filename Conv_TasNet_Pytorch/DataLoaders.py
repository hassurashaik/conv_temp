import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import soundfile as sf
from utils import handle_scp


def make_dataloader(
    is_train=True,
    data_kwargs=None,
    num_workers=0,
    chunk_size=24000,
    batch_size=16,
):
    dataset = Datasets(
        mix_scp=data_kwargs["mix_scp"],
        ref_scp=data_kwargs["ref_scp"],
        sr=data_kwargs["sr"],
        chunk_size=chunk_size,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=False,
    )


class Datasets(Dataset):
    """
    SCP-based fixed-chunk dataset for Conv-TasNet
    """

    def __init__(self, mix_scp, ref_scp, sr=8000, chunk_size=24000):
        super().__init__()

        self.mix_audio = handle_scp(mix_scp)
        self.ref_audio = [handle_scp(r) for r in ref_scp]

        self.sr = sr
        self.chunk_size = chunk_size

        # Keep only files long enough
        self.key = []
        for k, path in self.mix_audio.items():
            info = sf.info(path)
            if info.frames >= self.chunk_size:
                self.key.append(k)

        print(f"Loaded {len(self.key)} usable utterances")

    def __len__(self):
        return len(self.key)

    def __getitem__(self, idx):
        key = self.key[idx]

        mix_path = self.mix_audio[key]
        ref_paths = [r[key] for r in self.ref_audio]

        info = sf.info(mix_path)
        max_start = info.frames - self.chunk_size
        start = np.random.randint(0, max_start + 1)

        mix, _ = sf.read(
            mix_path,
            start=start,
            stop=start + self.chunk_size,
            dtype="float32",
        )

        refs = []
        for p in ref_paths:
          r, _ = sf.read(
        p,
        start=start,
        stop=start + self.chunk_size,
        dtype="float32",
          )
          refs.append(torch.from_numpy(r))

        refs = torch.stack(refs, dim=0)  # [C, T]

        return {
    "mix": torch.from_numpy(mix),   # [T]
    "ref": refs,                    # [C, T]
}




if __name__ == "__main__":
    datasets = Datasets('/content/conv_temp/Conv_TasNet_Pytorch/scp/cv_mix.scp',
                        ['/content/conv_temp/Conv_TasNet_Pytorch/scp/cv_s1.scp', '/content/conv_temp/Conv_TasNet_Pytorch/scp/cv_s2.scp'])
    print(datasets.key.index('012c020o_1.2887_409o0319_-1.2887.wav'))
