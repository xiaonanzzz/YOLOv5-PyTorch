# Copyright 2020 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import time

import torch
import torch.backends.cudnn
import torch.distributed
import torch.nn


def init_seeds(seed=0):
    torch.manual_seed(seed)

    if seed == 0:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def is_parallel(model):
    # is model is parallel with DP or DDP
    return type(model) in (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel)


def select_device(device="cuda", apex=True, batch_size=None):
    # device = "cpu" or "cuda:0"
    if device.lower() == "cpu":  # if device requested other than "cpu"
        print('Using cpus!!!!!!!!!')
        return torch.device("cpu")

    gpu_count = torch.cuda.device_count()

    if device == "cuda":
        # using multiple gpus
        assert batch_size % gpu_count == 0, f"batch size {batch_size} % {gpu_count} != 0"
        assert torch.cuda.is_available(), f"CUDA unavailable, invalid device {device} requested"

        return torch.device("cuda")

    print('single gpu mode!!!')
    assert int(device) < gpu_count, f'gpu index wrong: {device} >= {gpu_count}'
    return torch.device(f"cuda:{device}")


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()
