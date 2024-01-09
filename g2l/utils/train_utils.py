import os
import shutil
import time
import pickle

import numpy as np
import random
from copy import deepcopy

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from heapq import *
from .lr_scheduler import LinearWarmupMultiStepLR, LinearWarmupCosineAnnealingLR

def dijkstra(edge_indexs, edge_weight, s, N):
    N = 1000010
    inf = float("inf")
    graph = [-1] * N
    dis = [inf] * N
    e = [0] * N   # 给定边的idx，找出指向的节点号
    ne = [0] * N
    state = [0] * N
    weight = [0] * N # 给定边的idx，找出其权重大小
    global idx
    idx = 0
    def add(a,b,c):
        global idx
        idx += 1
        e[idx] = b
        ne[idx] = graph[a]
        weight[idx] = c
        graph[a] = idx

    def Prime_Dijkstra():
        heap = []
        heappush(heap,(0,s))

        while heap:
            distance,node = heappop(heap)
            if state[node]:
                continue
            else:
                state[node] = True

            cur = graph[node]
            while cur != -1:
                j = e[cur]
                if dis[j] > distance + weight[cur]:
                    dis[j] = weight[cur] + distance
                    heappush(heap,(dis[j],j))
                cur = ne[cur]
        
        return dis

    for i in edge_indexs:
        add(i[0], i[1], edge_weight[i[0]][i[1]])
    return Prime_Dijkstra()
    
    
def make_scheduler(
    optimizer,
    # optimizer_config,
    # num_iters_per_epoch,
    cfg,
    num_iters_per_epoch,
    last_epoch=-1
):
    """create scheduler
    return a supported scheduler
    All scheduler returned by this function should step every iteration
    """
    max_epoch = cfg.SOLVER.MAX_EPOCH+cfg.SOLVER.WARMUP_EPOCHS
    max_steps = max_epoch * num_iters_per_epoch

    warmup_steps = cfg.SOLVER.WARMUP_EPOCHS * num_iters_per_epoch
    steps = [num_iters_per_epoch * step for step in cfg.SOLVER.MILESTONES]

    if cfg.SOLVER.LR_SCHEDULER=="warmup_multistep":
        # Multi step
        # steps=cfg.SOLVER.MILESTONES
        scheduler = LinearWarmupMultiStepLR(
            optimizer,
            warmup_steps,
            steps,
            gamma=0.1,
            last_epoch=last_epoch
        )
    elif cfg.SOLVER.LR_SCHEDULER=="warmup_cosine":

        # Cosine
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_steps,
            max_steps,
            last_epoch=last_epoch
        )
    return scheduler

    # if optimizer_config["warmup"]:
    #     max_epochs = optimizer_config["epochs"] + optimizer_config["warmup_epochs"]
    #     max_steps = max_epochs * num_iters_per_epoch

    #     # get warmup params
    #     warmup_epochs = optimizer_config["warmup_epochs"]
    #     warmup_steps = warmup_epochs * num_iters_per_epoch

    #     # with linear warmup: call our custom schedulers
    #     if optimizer_config["schedule_type"] == "cosine":
    #         # Cosine
    #         scheduler = LinearWarmupCosineAnnealingLR(
    #             optimizer,
    #             warmup_steps,
    #             max_steps,
    #             last_epoch=last_epoch
    #         )

    #     elif optimizer_config["schedule_type"] == "multistep":
    #         # Multi step
    #         steps = [num_iters_per_epoch * step for step in optimizer_config["schedule_steps"]]
    #         scheduler = LinearWarmupMultiStepLR(
    #             optimizer,
    #             warmup_steps,
    #             steps,
    #             gamma=optimizer_config["schedule_gamma"],
    #             last_epoch=last_epoch
    #         )
    #     else:
    #         raise TypeError("Unsupported scheduler!")

    # else:
    #     max_epochs = optimizer_config["epochs"]
    #     max_steps = max_epochs * num_iters_per_epoch

    #     # without warmup: call default schedulers
    #     if optimizer_config["schedule_type"] == "cosine":
    #         # step per iteration
    #         scheduler = optim.lr_scheduler.CosineAnnealingLR(
    #             optimizer,
    #             max_steps,
    #             last_epoch=last_epoch
    #         )

    #     elif optimizer_config["schedule_type"] == "multistep":
    #         # step every some epochs
    #         steps = [num_iters_per_epoch * step for step in optimizer_config["schedule_steps"]]
    #         scheduler = optim.lr_scheduler.MultiStepLR(
    #             optimizer,
    #             steps,
    #             gamma=schedule_config["gamma"],
    #             last_epoch=last_epoch
    #         )
    #     else:
    #         raise TypeError("Unsupported scheduler!")

    # return scheduler


class AverageMeter(object):
    """Computes and stores the average and current value.
    Used to compute dataset stats from mini-batches
    """
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = 0.0

    def initialize(self, val, n):
        self.val = val
        self.avg = val
        self.sum = val * n
        self.count = n
        self.initialized = True

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.999, device=None):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)