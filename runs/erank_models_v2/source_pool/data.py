"""Official CIFAR data; train, checkpoint validation and rank calibration are disjoint."""
import concurrent.futures
import hashlib
import json
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / 'experiments_v2/config.json').read_text())
RUN = ROOT / 'runs' / CFG['protocol']


def download100():
    folder = ROOT / 'data/cifar100'
    folder.mkdir(parents=True, exist_ok=True)
    if all((folder / (name + '.npy')).exists() for name in
           ('train_images', 'train_labels', 'test_images', 'test_labels')):
        return
    url = 'https://cave.cs.toronto.edu/kriz/cifar-100-binary.tar.gz'
    md5 = '03b5dce01913d631647c71ecec9e9cb8'
    archive = folder / 'cifar-100-binary.tar.gz'
    if not archive.exists():
        request = urllib.request.Request(url, headers={'Range': 'bytes=0-0'})
        with urllib.request.urlopen(request, timeout=40) as response:
            if response.status != 206:
                raise RuntimeError('CIFAR server did not honor Range request')
            total = int(response.headers['Content-Range'].split('/')[-1])
        chunks = []
        chunk_size = (total + 11) // 12
        for start in range(0, total, chunk_size):
            end = min(total, start + chunk_size) - 1
            chunks.append((start, end, folder / f'part-{start}-{end}'))
        def fetch(chunk):
            import time
            start, end, path = chunk
            if path.exists() and path.stat().st_size == end - start + 1:
                return
            for attempt in range(5):
                try:
                    request = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
                    with urllib.request.urlopen(request, timeout=60) as response, path.open('wb') as out:
                        if response.status != 206:
                            raise RuntimeError('Range ignored')
                        shutil.copyfileobj(response, out, length=1024 * 1024)
                    assert path.stat().st_size == end - start + 1
                    print('Downloaded CIFAR-100 range', start, end, flush=True)
                    return
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(2)
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            list(executor.map(fetch, chunks))
        temporary = archive.with_suffix('.tmp')
        with temporary.open('wb') as out:
            for _, _, path in chunks:
                with path.open('rb') as inp:
                    shutil.copyfileobj(inp, out)
        assert temporary.stat().st_size == total
        assert hashlib.md5(temporary.read_bytes()).hexdigest() == md5
        temporary.replace(archive)
        for _, _, path in chunks:
            path.unlink()
    assert hashlib.md5(archive.read_bytes()).hexdigest() == md5
    with tarfile.open(archive, 'r:gz') as tar:
        for split in ('train', 'test'):
            stream = tar.extractfile(f'cifar-100-binary/{split}.bin')
            records = np.frombuffer(stream.read(), dtype=np.uint8).reshape(-1, 3074)
            np.save(folder / f'{split}_images.npy', records[:, 2:].reshape(-1, 3, 32, 32).copy())
            np.save(folder / f'{split}_labels.npy', records[:, 1].astype(np.int64))
    print('Verified official CIFAR-100 binary archive', md5, flush=True)


class Data:
    def __init__(self, dataset):
        self.dataset = dataset
        self.folder = ROOT / 'data' / ('cifar100' if dataset == 'cifar100' else '')
        self.classes = 100 if dataset == 'cifar100' else 10
        images = np.load(self.folder / 'train_images.npy')
        labels = np.load(self.folder / 'train_labels.npy')
        assert images.shape == (50000, 3, 32, 32)
        rng = np.random.default_rng(CFG['split_seed'])
        train, val, calibration = [], [], []
        count = 2500 // self.classes
        for label in range(self.classes):
            ids = rng.permutation(np.flatnonzero(labels == label))
            val.extend(ids[:count])
            calibration.extend(ids[count:2 * count])
            train.extend(ids[2 * count:])
        for indices in (train, val, calibration):
            rng.shuffle(indices)
        # Stratified prefixes: small diagnostic subsets also cover all classes.
        def balanced_order(indices, seed):
            random = np.random.default_rng(seed)
            by_class = [list(random.permutation([i for i in indices if labels[i] == c]))
                        for c in range(self.classes)]
            result = []
            while any(by_class):
                for c in random.permutation(self.classes):
                    if by_class[c]:
                        result.append(by_class[c].pop())
            return np.array(result, dtype=np.int64)
        calibration = balanced_order(calibration, 1701)
        # BN calibration is train-only, separate from scoring calibration.
        bn = balanced_order(train, 2701)[:1024]
        self.ids_cpu = {'train': np.array(train), 'val': np.array(val),
                        'calibration': calibration[:CFG['calibration_images']],
                        'calibration_holdout': calibration, 'bn': bn}
        assert not set(train).intersection(val)
        assert not set(train).intersection(calibration)
        assert not set(val).intersection(calibration)
        path = RUN / 'splits' / f'{dataset}.npz'
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            saved = np.load(path)
            assert all(np.array_equal(saved[k], v) for k, v in self.ids_cpu.items())
        else:
            temporary = path.with_suffix(f'.{os.getpid()}.tmp')
            with temporary.open('wb') as stream:
                np.savez(stream, **self.ids_cpu)
            temporary.replace(path)
        self.images = torch.from_numpy(images).cuda()
        self.labels = torch.from_numpy(labels).cuda()
        self.ids = {k: torch.as_tensor(v, device='cuda') for k, v in self.ids_cpu.items()}
        mean, std = ((0.4914, 0.4822, 0.4465), (0.247, 0.2435, 0.2616)) if self.classes == 10 else (
            (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        self.mean = torch.tensor(mean, device='cuda')[None, :, None, None]
        self.std = torch.tensor(std, device='cuda')[None, :, None, None]
        self.test_images = self.test_labels = None

    def batches(self, split, augment=False, seed=0, limit=None, batch_size=None):
        if split == 'test':
            if self.test_images is None:
                self.test_images = torch.from_numpy(np.load(self.folder / 'test_images.npy')).cuda()
                self.test_labels = torch.from_numpy(np.load(self.folder / 'test_labels.npy')).cuda()
            images, labels = self.test_images, self.test_labels
            ids = torch.arange(len(labels), device='cuda')
        else:
            images, labels, ids = self.images, self.labels, self.ids[split]
        if limit is not None:
            ids = ids[:limit]
        generator = torch.Generator(device='cuda').manual_seed(seed)
        if augment:
            ids = ids[torch.randperm(len(ids), generator=generator, device='cuda')]
        size = batch_size or CFG['batch_size' if augment else 'eval_batch_size']
        for start in range(0, len(ids), size):
            chosen = ids[start:start + size]
            x = images[chosen].float().div_(255)
            if augment:
                x = F.pad(x, (4, 4, 4, 4))
                offsets = torch.randint(0, 9, (len(chosen), 2), generator=generator, device='cuda')
                grid = torch.arange(32, device='cuda')
                rows, cols = offsets[:, 0, None] + grid, offsets[:, 1, None] + grid
                batch = torch.arange(len(chosen), device='cuda')[:, None, None]
                x = x.permute(0, 2, 3, 1)[batch, rows[:, :, None], cols[:, None, :]].permute(0, 3, 1, 2)
                flip = torch.rand((len(chosen), 1, 1, 1), generator=generator, device='cuda') < 0.5
                x = torch.where(flip, x.flip(-1), x)
            yield ((x - self.mean) / self.std).contiguous(memory_format=torch.channels_last), labels[chosen]


if __name__ == '__main__':
    download100()
