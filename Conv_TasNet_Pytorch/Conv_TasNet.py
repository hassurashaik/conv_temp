import torch
import torch.nn as nn

# =========================
# Normalization layers
# =========================
class GlobalLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, dim, 1))
        self.bias = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        mean = x.mean(dim=(1, 2), keepdim=True)
        var = (x - mean).pow(2).mean(dim=(1, 2), keepdim=True)
        return self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias


class CumulativeLayerNorm(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


def select_norm(norm, dim):
    if norm == "gln":
        return GlobalLayerNorm(dim)
    elif norm == "cln":
        return CumulativeLayerNorm(dim)
    else:
        return nn.BatchNorm1d(dim)

# =========================
# TCN Block
# =========================
class Conv1D_Block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, norm="gln", causal=False):
        super().__init__()

        self.conv1x1 = nn.Conv1d(in_ch, out_ch, 1)
        self.prelu1 = nn.PReLU()
        self.norm1 = select_norm(norm, out_ch)

        self.pad = dilation * (kernel_size - 1)
        self.dwconv = nn.Conv1d(
            out_ch,
            out_ch,
            kernel_size,
            dilation=dilation,
            padding=self.pad if causal else self.pad // 2,
            groups=out_ch,
        )

        self.prelu2 = nn.PReLU()
        self.norm2 = select_norm(norm, out_ch)
        self.res_conv = nn.Conv1d(out_ch, in_ch, 1)
        self.causal = causal

    def forward(self, x):
        y = self.conv1x1(x)
        y = self.prelu1(y)
        y = self.norm1(y)

        y = self.dwconv(y)
        if self.causal:
            y = y[:, :, :-self.pad]

        y = self.prelu2(y)
        y = self.norm2(y)
        y = self.res_conv(y)

        return x + y

# =========================
# STFT–TCN–iSTFT ConvTasNet
# =========================
class ConvTasNet(nn.Module):
    def __init__(
        self,
        n_fft=256,
        hop_length=64,
        B=96,        # bottleneck channels
        H=256,       # TCN hidden channels
        P=3,
        X=7,         # blocks per repeat
        R=3,         # repeats
        num_spks=2,
        norm="gln",
        causal=False,
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop = hop_length
        self.num_spks = num_spks
        self.freq_bins = n_fft // 2 + 1

        # STFT window
        self.register_buffer("window", torch.hann_window(n_fft))

        # Bottleneck
        self.layer_norm = GlobalLayerNorm(self.freq_bins)

        self.bottleneck = nn.Conv1d(self.freq_bins, B, 1)

        # TCN
        tcn_blocks = []
        for _ in range(R):
            for i in range(X):
                tcn_blocks.append(
                    Conv1D_Block(
                        B,
                        H,
                        kernel_size=P,
                        dilation=2 ** i,
                        norm=norm,
                        causal=causal,
                    )
                )
        self.tcn = nn.Sequential(*tcn_blocks)

        # Mask generator
        self.mask_conv = nn.Conv1d(B, num_spks * self.freq_bins, 1)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # ===== STFT =====
        X = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.window,
            return_complex=True,
            center=True,
        )

        mag = X.abs()           # [B, F, T]
        phase = X.angle()       # [B, F, T]

        # ===== TCN =====
        y = self.layer_norm(mag)
        y = self.bottleneck(y)
        y = self.tcn(y)

        masks = self.mask_conv(y)
        masks = torch.chunk(masks, self.num_spks, dim=1)
        masks = [torch.sigmoid(m) for m in masks]

        # ===== iSTFT =====
        outputs = []
        for m in masks:
            est_mag = m * mag
            est_complex = torch.polar(est_mag, phase)
            wav = torch.istft(
                est_complex,
                n_fft=self.n_fft,
                hop_length=self.hop,
                window=self.window,
                length=x.shape[-1],
                center=True,
            )
            outputs.append(wav)

        return outputs

# =========================
# Utility
# =========================
def check_parameters(net):
    return sum(p.numel() for p in net.parameters()) / 1e6


# =========================
# Test
# =========================
if __name__ == "__main__":
    x = torch.randn(1, 24000)
    model = ConvTasNet()
    y = model(x)
    print("Sources:", len(y))
    print("Output shape:", y[0].shape)
    print("Params (M):", check_parameters(model))
