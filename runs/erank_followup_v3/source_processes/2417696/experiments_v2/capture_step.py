"""Capture unchanged forward/backward work to reduce Python launch overhead.

The optimizer and learning-rate schedule remain outside the graph. BatchNorm buffers
are restored after warmup/capture, and separate batch-size graphs share gradient
buffers. No padded samples or changed batch statistics are introduced.
"""
import torch
from torch import nn
from torch.nn import functional as F

from data import CFG
from models import sparse_parameters


class CapturedStep:
    def __init__(self, model, optimizer, x, y, sparse=False):
        self.model, self.optimizer = model, optimizer
        self.is_vit = model.spec['family'] == 'mini_vit'
        self.sparse = sparse
        self.x, self.y, self.y2 = torch.empty_like(x), torch.empty_like(y), torch.empty_like(y)
        self.lam = torch.ones((), device='cuda')
        self.x.copy_(x), self.y.copy_(y), self.y2.copy_(y)
        # The same grad buffers must be used by both the full and last-batch graphs.
        for parameter in model.parameters():
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
        buffers = [(buffer, buffer.clone()) for buffer in model.buffers()]
        model.train()
        warmup = torch.cuda.Stream()
        warmup.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup):
            for _ in range(3):
                self.operation()
        torch.cuda.current_stream().wait_stream(warmup)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.operation()
        for destination, source in buffers:
            destination.copy_(source)
        optimizer.zero_grad(set_to_none=False)

    def operation(self):
        self.optimizer.zero_grad(set_to_none=False)
        with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
            logits = self.model(self.x)
            if self.is_vit:
                loss = self.lam * F.cross_entropy(logits, self.y, label_smoothing=CFG['vit_label_smoothing'])
                loss = loss + (1 - self.lam) * F.cross_entropy(logits, self.y2, label_smoothing=CFG['vit_label_smoothing'])
            else:
                loss = F.cross_entropy(logits, self.y)
            if self.sparse:
                penalty = sum(p.abs().sum() for p in sparse_parameters(self.model))
                loss = loss + CFG['gate_l1' if self.is_vit else 'bn_l1'] * penalty
        loss.backward()
        if self.is_vit:
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.loss = loss

    def __call__(self, x, y, y2=None, lam=1.0):
        self.x.copy_(x)
        self.y.copy_(y)
        if self.is_vit:
            self.y2.copy_(y2)
            self.lam.fill_(lam)
        self.graph.replay()
        self.optimizer.step()
        return self.loss
