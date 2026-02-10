import torch
from itertools import permutations


def si_snr(est, ref, eps=1e-8):
    """
    est, ref: [B, T]
    return: [B]
    """
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)

    proj = torch.sum(est * ref, dim=-1, keepdim=True) * ref \
           / (torch.sum(ref ** 2, dim=-1, keepdim=True) + eps)

    noise = est - proj
    ratio = torch.sum(proj ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + eps)

    return 10 * torch.log10(ratio + eps)


def pit_si_snr_loss(ests, refs):
    """
    ests, refs: [B, C, T]
    return: scalar loss
    """
    B, C, T = ests.shape

    perms = list(permutations(range(C)))
    scores = []

    for p in perms:
        s = 0
        for i, j in enumerate(p):
            s += si_snr(ests[:, i], refs[:, j])
        scores.append(s / C)

    scores = torch.stack(scores, dim=1)  # [B, P]
    max_score, _ = torch.max(scores, dim=1)

    return -max_score.mean()
