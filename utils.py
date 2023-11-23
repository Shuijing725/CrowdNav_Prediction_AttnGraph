def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    # random.seed(seed)
    # os.environ["PYTHONHASHSEED"] = str(seed)
    # np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.mps.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)
    # torch.mps.manual_seed_all(seed)


def get_device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"
