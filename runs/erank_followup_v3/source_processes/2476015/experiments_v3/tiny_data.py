"""Official Tiny ImageNet archive, deterministic splits, GPU-resident minibatches."""
import concurrent.futures
import hashlib
import io
import json
import os
from pathlib import Path
import time
import threading
import urllib.request
import zipfile

from runtime import CFG, ROOT, RUN, save_json
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

FOLDER = ROOT / 'data/tinyimagenet'
URL = 'https://cs231n.stanford.edu/tiny-imagenet-200.zip'
MD5 = '90528d7ca1a48142e341f4ef8d21d0de'


def checksum(path, algorithm='sha256'):
    digest = hashlib.new(algorithm)
    with path.open('rb') as stream:
        for piece in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(piece)
    return digest.hexdigest()


def download():
    FOLDER.mkdir(parents=True, exist_ok=True)
    archive = FOLDER / 'tiny-imagenet-200.zip'
    if not archive.exists():
        with urllib.request.urlopen(urllib.request.Request(URL, method='HEAD'), timeout=30) as response:
            total = int(response.headers['Content-Length'])
        chunk = (total + 11) // 12
        chunks = [(start,min(total-1,start+chunk-1),FOLDER/f'part-{start}') for start in range(0,total,chunk)]
        def fetch(item):
            start,end,path = item
            if path.exists() and path.stat().st_size == end-start+1:
                return
            for attempt in range(4):
                try:
                    request = urllib.request.Request(URL, headers={'Range':f'bytes={start}-{end}'})
                    with urllib.request.urlopen(request,timeout=60) as response, path.open('wb') as out:
                        assert response.status == 206
                        while piece := response.read(1024**2):
                            out.write(piece)
                    assert path.stat().st_size == end-start+1
                    print('Downloaded Tiny ImageNet range',start,end,flush=True)
                    return
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(1)
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            list(executor.map(fetch,chunks))
        temporary = archive.with_suffix('.tmp')
        with temporary.open('wb') as out:
            for _,_,path in chunks:
                with path.open('rb') as stream:
                    while piece := stream.read(1024**2):
                        out.write(piece)
        assert checksum(temporary,'md5') == MD5
        temporary.replace(archive)
        for _,_,path in chunks:
            path.unlink()
    assert checksum(archive,'md5') == MD5
    marker = FOLDER / 'decoded.json'
    if marker.exists():
        print('Verified Tiny ImageNet archive; decoded data already present',flush=True)
        return
    with zipfile.ZipFile(archive) as z:
        names = sorted(z.read('tiny-imagenet-200/wnids.txt').decode().splitlines())
        classes = {name:i for i,name in enumerate(names)}
        train_names = sorted(name for name in z.namelist() if '/train/' in name and '/images/' in name and name.endswith('.JPEG'))
        annotations = {line.split('\t')[0]:line.split('\t')[1] for line in z.read('tiny-imagenet-200/val/val_annotations.txt').decode().splitlines()}
        test_names = sorted(name for name in z.namelist() if '/val/images/' in name and name.endswith('.JPEG'))
        assert len(train_names)==100000 and len(test_names)==10000 and len(classes)==200
        archive_lock=threading.Lock()
        def decode(name):
            # Serialize archive seeks; Pillow decoding remains parallel.
            with archive_lock:
                payload=z.read(name)
            with Image.open(io.BytesIO(payload)) as im:
                arr = np.asarray(im.convert('RGB'),dtype=np.uint8)
            assert arr.shape==(64,64,3)
            return arr.transpose(2,0,1)
        for split,files in [('train',train_names),('test',test_names)]:
            destination = FOLDER / f'{split}_images.npy'
            images = np.lib.format.open_memmap(destination,mode='w+',dtype=np.uint8,shape=(len(files),3,64,64))
            labels = np.array([classes[name.split('/')[2] if split=='train' else annotations[Path(name).name]] for name in files],dtype=np.int64)
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                for i,arr in enumerate(executor.map(decode,files)):
                    images[i]=arr
                    if (i+1)%20000==0:
                        print('Decoded',split,i+1,flush=True)
            images.flush()
            del images
            np.save(FOLDER / f'{split}_labels.npy',labels)
            (FOLDER / f'{split}_filenames.json').write_text(json.dumps(files))
        (FOLDER/'classes.json').write_text(json.dumps(names))
    save_json(marker, {'archive_url':URL,'archive_md5':MD5,'archive_sha256':checksum(archive),
                      'official_validation_used_as_final_test':True,'created_at':time.time(),
                      'arrays_sha256':{p.name:checksum(p) for p in FOLDER.glob('*.npy')}})
    print('Tiny ImageNet decoded and checksummed',flush=True)


class TinyData:
    def __init__(self):
        assert (FOLDER/'decoded.json').exists(), 'Run tiny_data.py first'
        self.dataset,self.classes = 'tinyimagenet',200
        self.folder = FOLDER
        labels = np.load(FOLDER/'train_labels.npy')
        rng = np.random.default_rng(CFG['split_seed'])
        parts = {'train':[],'val':[],'calibration_holdout':[]}
        for c in range(self.classes):
            ids = rng.permutation(np.flatnonzero(labels==c))
            parts['val'].extend(ids[:20]);parts['calibration_holdout'].extend(ids[20:40]);parts['train'].extend(ids[40:])
        for name in parts:
            parts[name] = np.array(parts[name],dtype=np.int64)
            rng.shuffle(parts[name])
        assert len(parts['train'])==92000
        assert not set(parts['train']).intersection(parts['val'])
        assert not set(parts['calibration_holdout']).intersection(parts['train'])
        assert not set(parts['calibration_holdout']).intersection(parts['val'])
        parts['bn'] = parts['train'][:1024]
        parts['calibration'] = parts['calibration_holdout'][:1024]
        self.ids_cpu = parts
        path = RUN/'splits/tinyimagenet.npz'
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            old=np.load(path)
            assert all(np.array_equal(old[k],v) for k,v in parts.items())
        else:
            with path.with_suffix(f'.{os.getpid()}.tmp').open('wb') as f:
                np.savez(f,**parts)
            path.with_suffix(f'.{os.getpid()}.tmp').replace(path)
        self.images = torch.from_numpy(np.load(FOLDER/'train_images.npy')).cuda()
        self.labels = torch.from_numpy(labels).cuda()
        self.ids = {k:torch.as_tensor(v,device='cuda') for k,v in parts.items()}
        self.mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
        self.std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
        self.test_images=self.test_labels=None

    def batches(self,split,augment=False,seed=0,limit=None,batch_size=None):
        if split=='test':
            if self.test_images is None:
                self.test_images=torch.from_numpy(np.load(FOLDER/'test_images.npy')).cuda()
                self.test_labels=torch.from_numpy(np.load(FOLDER/'test_labels.npy')).cuda()
            images,labels=self.test_images,self.test_labels
            ids=torch.arange(len(labels),device='cuda')
        else:
            images,labels,ids=self.images,self.labels,self.ids[split]
        if limit is not None:
            ids=ids[:limit]
        generator=torch.Generator(device='cuda').manual_seed(seed)
        if augment:
            ids=ids[torch.randperm(len(ids),generator=generator,device='cuda')]
        size=batch_size or (256 if augment else 256)
        for start in range(0,len(ids),size):
            chosen=ids[start:start+size]
            x=images[chosen].float().div_(255)
            if augment:
                x=F.pad(x,(8,8,8,8))
                offsets=torch.randint(0,17,(len(chosen),2),generator=generator,device='cuda')
                grid=torch.arange(64,device='cuda')
                rows,cols=offsets[:,0,None]+grid,offsets[:,1,None]+grid
                batch=torch.arange(len(chosen),device='cuda')[:,None,None]
                x=x.permute(0,2,3,1)[batch,rows[:,:,None],cols[:,None,:]].permute(0,3,1,2)
                flip=torch.rand((len(chosen),1,1,1),generator=generator,device='cuda')<.5
                x=torch.where(flip,x.flip(-1),x)
            yield ((x-self.mean)/self.std).contiguous(memory_format=torch.channels_last),labels[chosen]


if __name__=='__main__':
    download()
