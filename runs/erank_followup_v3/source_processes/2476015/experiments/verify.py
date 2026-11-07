"""Scientific invariants: physical pruning, reconstruction, and matched budgets."""
import numpy as np
import torch

from common import CONFIG, MAC_WEIGHTS, WIDTHS, ResNet20, model_cost, on_gpu, setup
from methods import entropy_rank, fold_deleted_channels, global_random_widths, prune, widths_at_budget


def verify_costs():
    torch.set_num_threads(4)
    for budget in CONFIG["flops_retained"]:
        widths = widths_at_budget(WIDTHS, budget)
        model = ResNet20(widths).eval()
        observed = [0]
        def count(layer, inputs, output):
            if isinstance(layer, torch.nn.Conv2d):
                observed[0] += output.numel() * layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]
            elif isinstance(layer, torch.nn.Linear):
                observed[0] += output.numel() * layer.in_features
        handles = [m.register_forward_hook(count) for m in model.modules()
                   if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
        with torch.inference_mode():
            model(torch.zeros(1, 3, 32, 32))
        for handle in handles:
            handle.remove()
        assert observed[0] == model_cost(model)["macs"], (budget, observed, model_cost(model))
    print("PASS: analytical budgets equal MAC counts from actual pruned forward passes")


def main():
    verify_costs()
    setup()
    assert abs(float(entropy_rank([1, 1, 1])) - 3) < 1e-12
    assert abs(float(entropy_rank([4, 0, 0])) - 1) < 1e-12
    assert float(entropy_rank([0, 0])) == 0
    for retained in CONFIG["flops_retained"]:
        targets = []
        for profile in (WIDTHS, np.array(WIDTHS) * np.linspace(0.4, 0.95, 9),
                        np.array(WIDTHS) * np.linspace(0.85, 0.5, 9)):
            widths = widths_at_budget(profile, retained)
            targets.append(int(MAC_WEIGHTS @ widths))
            assert all(4 <= w <= c for w, c in zip(widths, WIDTHS))
        assert len(set(targets)) == 1
        removed = sum(WIDTHS) - sum(widths)
        random_widths, attempts = global_random_widths(removed, 123, targets[0])
        assert sum(random_widths) == sum(widths)
        assert MAC_WEIGHTS @ random_widths == targets[0]
    # Exact duplicate channels must admit weight folding; plain deletion need not.
    z = np.random.default_rng(7).normal(size=(1000, 1))
    x = np.concatenate([z, z], axis=1)
    weight = np.array([[[[1.]], [[2.]]]], dtype=np.float32)
    folded = fold_deleted_channels(weight, x, np.array([0]))
    assert abs(float(folded[0, 0, 0, 0]) - 3) < 1e-5
    model = on_gpu(ResNet20()).eval()
    inputs = torch.randn(4, 3, 32, 32, device="cuda")
    with torch.inference_mode():
        output = model(inputs)
        identity, _ = prune(model, WIDTHS, [list(range(c)) for c in WIDTHS])
        identity.eval()
        assert torch.allclose(output, identity(inputs), atol=1e-5, rtol=1e-5)
        # Functional invariance under positive BN scaling and inverse conv2 scaling.
        block = identity.blocks[4]
        scale = torch.exp(torch.linspace(-2, 2, 32, device="cuda"))
        block.bn1.weight.mul_(scale)
        block.bn1.bias.mul_(scale)
        block.conv2.weight.div_(scale[None, :, None, None])
        assert torch.allclose(output, identity(inputs), atol=2e-4, rtol=2e-4)
    print("PASS: rank identities; exact matched MAC budgets; global Random constraints; "
          "duplicate-channel reconstruction; full-width pruning identity; rescaling invariance")


if __name__ == "__main__":
    main()
