# ERank model and dataset extension

Read [`experiment_pool_v2.md`](../experiment_pool_v2.md) for the frozen scientific protocol. The implementation and artifacts remain local; no publication is authorized at this time.

## Execution

```bash
python3 experiments/download_data.py
python3 experiments_v2/data.py
python3 experiments_v2/manifest.py
python3 experiments_v2/launch.py verify
python3 experiments_v2/launch.py base
python3 experiments_v2/controller.py
```

The controller continues each of the four model/dataset groups into its pool after that group's base training ends. It starts held-out testing only when all 504 trainings have finished, followed by exclusive GPU 0 latency measurement and reporting. The controller also inherits or explicitly sets the GPU UUID for child processes. Run it in a persistent terminal, or launch via a detached subprocess. It requires `experiments_v2/launch.py base` to have created `processes.json`.

`launch.py pool` is a manual alternative when all base workers have exited. `launch.py finalize` resumes the final phases without repeating completed model evaluations or timing rounds. Existing completed trainings are reused; incomplete jobs resume from `last.pt`. A worker failure is recorded under `runs/erank_models_v2/failures/` and is not silently treated as a result.

The first base worker processes use the original eager training implementation saved in `source_base/`. Later training can capture the same forward/backward work in CUDA graphs to reduce Python launch overhead. Optimizers, learning rates, precision, batch sizes, augmentation, and sample counts remain unchanged. `verify_capture.py` compares full parameters and buffers after eight optimizer updates for both families, with and without sparsity, alternating two batch sizes; the maximum observed difference was exactly zero. Capture warmup restores all BN buffers; the final short batch has its own graph and is never padded. Set `ERANK_USE_CUDA_GRAPH=0` for the original eager implementation. Inference latency is always measured in eager mode.

The full training engine was additionally compared for two complete 45,000-image epochs in both families, including augmentation, Mixup, checkpointing, and the final short minibatch. All final parameters and buffers were bitwise identical (`verification_full_engine.json`).

## Main code

- `data.py`: checksum-verified official data and disjoint train/validation/calibration splits.
- `models.py`: explicit ResNet-20, Mini-ViT, and factorized modules with serializable architectures.
- `engine.py`: reproducible training, recovery, checkpoint selection, rank trajectories.
- `measures.py`: pooled spectral ranks, pivoted QR, L1, per-example Taylor, spatial HRank.
- `compress.py`: exact channel budgets, physical slicing, attention heads, SVD, alternative ViT architectures.
- `pool.py`: main comparisons, controls, scratch runs, iterations and diagnostics.
- `finalize.py`: manifest validation, test predictions and exclusive latency measurements.
- `report.py`: per-run CSV, paired confidence intervals, tables and exportable plots.
- `rank_preview.py`: plots all completed dense rank trajectories without opening test data.
- `monitor.py`: live progress and explicit worker failure records.

The final report also writes `findings.md` with the actual paired ERank/Uniform results,
inference speedups, and the ViT architecture table. Both reports require all training,
test evaluation, and latency series to finish; they never substitute validation for test.
Statistical clarifications made before test are recorded in `analysis_amendments.json`.

No torchvision, timm, external pretrained models, or GitHub publishing is required.
