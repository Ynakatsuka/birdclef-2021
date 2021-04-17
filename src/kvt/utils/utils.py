from contextlib import contextmanager
import math
import os
import random
import psutil
import sys
import time

import numpy as np
import torch


def seed_torch(seed=None, random_seed=True):
    if random_seed or seed is None:
        seed = np.random.randint(0, 1000000)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


@contextmanager
def trace(title, logger=None):
    t0 = time.time()
    p = psutil.Process(os.getpid())
    m0 = p.memory_info()[0] / 2. ** 30
    yield
    m1 = p.memory_info()[0] / 2. ** 30
    delta = m1 - m0
    sign = '+' if delta >= 0 else '-'
    delta = math.fabs(delta)
    message = f'[{m1:.1f}GB({sign}{delta:.1f}GB):{time.time() - t0:.1f}sec] {title} '
    print(message)
    if logger is not None:
        logger.info(message)
