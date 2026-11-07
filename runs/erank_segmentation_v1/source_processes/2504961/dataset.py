"""Fixed ADE20K splits; identical preprocessing for dense and compressed models."""
import json
from pathlib import Path

from common import CFG, ROOT, RUN, save_json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset,DataLoader,Sampler

FOLDER=ROOT/'data/ade20k/ADEChallengeData2016'
MEAN=torch.tensor([.485,.456,.406])[:,None,None]
STD=torch.tensor([.229,.224,.225])[:,None,None]


def manifest():
    destination=RUN/'split.json'
    if destination.exists():return json.loads(destination.read_text())
    assert (ROOT/'data/ade20k/extracted.json').exists(),'Prepare the dataset first'
    filenames=sorted(p.name for p in (FOLDER/'images/training').glob('*.jpg'))
    assert len(filenames)==20210
    order=np.random.default_rng(CFG['split_seed']).permutation(len(filenames))
    a=CFG['calibration_pool'];b=a+CFG['checkpoint_validation']
    result={name:[filenames[i] for i in ids] for name,ids in [('calibration',order[:a]),('validation',order[a:b]),('train',order[b:])]}
    result['test']=sorted(p.name for p in (FOLDER/'images/validation').glob('*.jpg'))
    assert len(result['test'])==2000
    assert not set(result['train']).intersection(result['validation'])
    assert not set(result['calibration']).intersection(result['train'])
    assert not set(result['calibration']).intersection(result['validation'])
    save_json(destination,result)
    return result


def reduce_labels(mask):
    original=np.asarray(mask,dtype=np.uint8)
    assert original.max()<=150
    reduced=original.astype(np.int64)-1
    reduced[original==0]=255
    return reduced


class ADE(Dataset):
    def __init__(self,split,filenames=None,augment=False,original_masks=False):
        self.names=filenames if filenames is not None else manifest()[split]
        self.split=split;self.folder='validation' if split=='test' else 'training'
        self.augment=augment;self.original_masks=original_masks

    def __len__(self):return len(self.names)

    def __getitem__(self,index):
        flip=False
        if isinstance(index,tuple):index,flip=index
        name=self.names[index]
        with Image.open(FOLDER/'images'/self.folder/name) as image:
            original_size=image.size
            image=image.convert('RGB').resize((CFG['image_size'],CFG['image_size']),Image.Resampling.BILINEAR)
            array=np.asarray(image,dtype=np.uint8).copy()
        with Image.open(FOLDER/'annotations'/self.folder/(Path(name).stem+'.png')) as image:
            assert image.size==original_size
            if not self.original_masks:image=image.resize((CFG['image_size'],CFG['image_size']),Image.Resampling.NEAREST)
            mask=reduce_labels(image)
        if flip:
            array=array[:,::-1].copy();mask=mask[:,::-1].copy()
        x=torch.from_numpy(array.transpose(2,0,1).copy()).float()/255
        return (x-MEAN)/STD,torch.from_numpy(mask),name


def evaluation_collate(items):
    x,y,names=zip(*items)
    return torch.stack(x),list(y),list(names)


def evaluation_loader(split,filenames=None,batch_size=4,workers=4):
    data=ADE(split,filenames=filenames,original_masks=True)
    return DataLoader(data,batch_size=batch_size,shuffle=False,num_workers=workers,pin_memory=True,
                      collate_fn=evaluation_collate,persistent_workers=workers>0)


class StepSampler(Sampler):
    def __init__(self,size,seed,start=0,steps=None):
        self.size=size;self.seed=seed;self.start=start;self.steps=steps or CFG['steps']

    def __len__(self):return self.steps-self.start

    def __iter__(self):
        rng=np.random.default_rng(self.seed);flip_rng=np.random.default_rng(self.seed+9101)
        permutation=rng.permutation(self.size);offset=0
        for step in range(self.steps):
            if offset+CFG['batch_size']>self.size:
                permutation=rng.permutation(self.size);offset=0
            ids=permutation[offset:offset+CFG['batch_size']];offset+=CFG['batch_size']
            flips=flip_rng.random(len(ids))<.5
            if step>=self.start:yield [(int(i),bool(f)) for i,f in zip(ids,flips)]


def training_loader(seed,start=0,workers=4):
    data=ADE('train',augment=True)
    return DataLoader(data,batch_sampler=StepSampler(len(data),seed,start),num_workers=workers,
                      pin_memory=True,persistent_workers=workers>0)
