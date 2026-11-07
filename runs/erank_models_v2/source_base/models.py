"""Small, structurally compressible CNN and ViT; explicit architecture serialization."""
import copy
import math
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from common import Block

CNN_WIDTHS = [16, 16, 16, 32, 32, 32, 64, 64, 64]


class ResNet(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = copy.deepcopy(spec)
        widths = spec.get('widths', CNN_WIDTHS)
        self.spec['widths'] = list(widths)
        self.conv = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.blocks = nn.ModuleList()
        cin = 16
        for i, width in enumerate(widths):
            self.blocks.append(Block(cin, CNN_WIDTHS[i], width, 2 if i in (3, 6) else 1))
            cin = CNN_WIDTHS[i]
        self.fc = nn.Linear(64, spec['classes'])
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        for block in self.blocks:
            x = block(x)
        return self.fc(x.mean((2, 3)))


class Attention(nn.Module):
    def __init__(self, dim, heads, head_dim):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        inner = heads * head_dim
        self.qkv = nn.Linear(dim, 3 * inner)
        self.proj = nn.Linear(inner, dim)

    def forward(self, x):
        b, n, _ = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # Explicit attention makes deterministic behavior and MAC counting transparent.
        weights = (q @ k.transpose(-2, -1) * self.head_dim ** -0.5).softmax(-1)
        context = (weights @ v).transpose(1, 2).reshape(b, n, -1)
        return self.proj(context)


class MLP(nn.Module):
    def __init__(self, dim, width, gated):
        super().__init__()
        self.fc1 = nn.Linear(dim, width)
        self.fc2 = nn.Linear(width, dim)
        if gated:
            self.gate = nn.Parameter(torch.ones(width))
        else:
            self.register_buffer('gate', torch.ones(width))

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)) * self.gate)


class TransformerBlock(nn.Module):
    def __init__(self, dim, width, heads, head_dim, gated):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, head_dim)
        self.mlp = MLP(dim, width, gated)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class MiniViT(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = copy.deepcopy(spec)
        d, p, depth = spec['dim'], spec['patch_size'], spec['depth']
        widths = spec.get('widths', [spec.get('mlp_width', 384)] * depth)
        heads = spec.get('head_counts', [spec['heads']] * depth)
        self.spec.update(widths=list(widths), head_counts=list(heads))
        self.patch = nn.Conv2d(3, d, p, stride=p)
        self.pos = nn.Parameter(torch.zeros(1, (32 // p) ** 2, d))
        self.blocks = nn.ModuleList([TransformerBlock(d, widths[i], heads[i], spec['head_dim'],
                                                     spec.get('gated', False)) for i in range(depth)])
        self.norm = nn.LayerNorm(d)
        self.fc = nn.Linear(d, spec['classes'])
        nn.init.trunc_normal_(self.pos, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.patch(x).flatten(2).transpose(1, 2) + self.pos
        for block in self.blocks:
            x = block(x)
        return self.fc(self.norm(x).mean(1))


class FactorLinear(nn.Module):
    def __init__(self, source, rank):
        super().__init__()
        self.a = nn.Linear(source.in_features, rank, bias=False)
        self.b = nn.Linear(rank, source.out_features, bias=source.bias is not None)

    def forward(self, x):
        return self.b(self.a(x))


class FactorConv(nn.Module):
    def __init__(self, source, rank):
        super().__init__()
        self.a = nn.Conv2d(source.in_channels, rank, source.kernel_size, source.stride,
                           source.padding, dilation=source.dilation, bias=False)
        self.b = nn.Conv2d(rank, source.out_channels, 1, bias=source.bias is not None)

    def forward(self, x):
        return self.b(self.a(x))


def replace_module(model, name, module):
    parent_name, _, leaf = name.rpartition('.')
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, leaf, module)


def build(spec):
    model = ResNet(spec) if spec['family'] == 'resnet20' else MiniViT(spec)
    for name, rank in spec.get('factors', {}).items():
        old = model.get_submodule(name)
        replacement = FactorConv(old, rank) if isinstance(old, nn.Conv2d) else FactorLinear(old, rank)
        replace_module(model, name, replacement)
    return model


def base_spec(family, classes, vit, gated=False):
    if family == 'resnet20':
        return {'family': family, 'classes': classes, 'widths': CNN_WIDTHS.copy()}
    return {'family': family, 'classes': classes, **vit, 'gated': gated}


def feature_modules(model):
    if model.spec['family'] == 'resnet20':
        return [block.conv2 for block in model.blocks]
    return [block.mlp.fc2 for block in model.blocks]


def sparse_parameters(model):
    if model.spec['family'] == 'resnet20':
        return [block.bn1.weight for block in model.blocks]
    return [block.mlp.gate for block in model.blocks]


def to_gpu(model):
    return model.to('cuda', memory_format=torch.channels_last)


@torch.inference_mode()
def cost(model):
    """Count Conv/Linear MACs from actual module shapes; add both attention products."""
    macs = 0
    handles = []
    def hook(module, inputs, output):
        nonlocal macs
        if isinstance(module, nn.Conv2d):
            macs += output.numel() * module.in_channels * math.prod(module.kernel_size) // module.groups
        elif isinstance(module, nn.Linear):
            macs += output.numel() * module.in_features
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(hook))
    training = model.training
    model.eval()
    try:
        model(torch.zeros(1, 3, 32, 32, device=next(model.parameters()).device))
    finally:
        for handle in handles:
            handle.remove()
        model.train(training)
    if model.spec['family'] == 'mini_vit':
        tokens = (32 // model.spec['patch_size']) ** 2
        macs += sum(2 * tokens ** 2 * b.attn.heads * b.attn.head_dim for b in model.blocks)
    return {'params': sum(p.numel() for p in model.parameters()), 'macs': int(macs), 'flops': int(2 * macs)}
