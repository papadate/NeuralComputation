import torch


def mps_checker():
    if torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'
