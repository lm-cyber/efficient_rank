"""Download official ADE20K and a revision-pinned NVIDIA checkpoint without tokens."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/erank_segmentation_v1'
DATA=ROOT/'data/ade20k'
WEIGHTS=ROOT/'data/segformer_b1'
DATA_URL='https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip'
MODEL='nvidia/segformer-b1-finetuned-ade-512-512'
REVISION='a775b4f5dcb2179e57006dca0e268281306412ef'


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def fetch_file(url,destination):
    if destination.exists():return
    temporary=destination.with_suffix(destination.suffix+'.download')
    with urllib.request.urlopen(url,timeout=90) as response,temporary.open('wb') as out:
        while chunk:=response.read(1024**2):out.write(chunk)
    temporary.replace(destination)
    print('Downloaded',destination.name,flush=True)


def model():
    WEIGHTS.mkdir(parents=True,exist_ok=True)
    for name in ('config.json','preprocessor_config.json','pytorch_model.bin','README.md'):
        fetch_file(f'https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}',WEIGHTS/name)
    (RUN/'model_source.json').write_text(json.dumps({'model':MODEL,'revision':REVISION,'at':time.time(),
        'sha256':{p.name:sha256(p) for p in WEIGHTS.iterdir() if p.is_file()}},indent=2))
    print('Pinned model downloaded',flush=True)


def dataset():
    DATA.mkdir(parents=True,exist_ok=True)
    archive=DATA/'ADEChallengeData2016.zip'
    if not archive.exists():
        with urllib.request.urlopen(urllib.request.Request(DATA_URL,method='HEAD'),timeout=30) as response:
            size=int(response.headers['Content-Length'])
        part_size=(size+15)//16
        parts=[(start,min(size-1,start+part_size-1),DATA/f'part-{start}') for start in range(0,size,part_size)]
        def part(item):
            start,end,path=item
            if path.exists() and path.stat().st_size==end-start+1:return
            for attempt in range(4):
                try:
                    req=urllib.request.Request(DATA_URL,headers={'Range':f'bytes={start}-{end}'})
                    with urllib.request.urlopen(req,timeout=120) as response,path.open('wb') as out:
                        assert response.status==206
                        assert response.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
                        while chunk:=response.read(1024**2):out.write(chunk)
                    assert path.stat().st_size==end-start+1
                    print('ADE range complete',start,end,flush=True)
                    return
                except Exception:
                    if attempt==3:raise
                    time.sleep(2)
        with ThreadPoolExecutor(max_workers=16) as executor:list(executor.map(part,parts))
        temporary=archive.with_suffix('.tmp')
        with temporary.open('wb') as out:
            for _,_,path in parts:
                with path.open('rb') as stream:
                    while chunk:=stream.read(1024**2):out.write(chunk)
        assert temporary.stat().st_size==size
        temporary.replace(archive)
        for _,_,path in parts:path.unlink()
    if not (DATA/'extracted.json').exists():
        extracted=0
        # Serialize ZIP reads: no shared seek state across decoder workers.
        with zipfile.ZipFile(archive) as zipped:
            for entry in zipped.infolist():
                path=DATA/entry.filename
                assert path.resolve().is_relative_to(DATA.resolve())
                if entry.is_dir():path.mkdir(parents=True,exist_ok=True);continue
                path.parent.mkdir(parents=True,exist_ok=True)
                payload=zipped.read(entry)  # Includes per-entry CRC validation.
                path.write_bytes(payload);extracted+=1
        root=DATA/'ADEChallengeData2016'
        counts={name:len(list((root/'images'/name).glob('*.jpg'))) for name in ('training','validation')}
        assert counts=={'training':20210,'validation':2000},counts
        assert all((root/'annotations'/name/(p.stem+'.png')).exists() for name in counts for p in (root/'images'/name).glob('*.jpg'))
        (DATA/'extracted.json').write_text(json.dumps({'url':DATA_URL,'sha256':sha256(archive),
            'counts':counts,'files_crc_checked':extracted,'at':time.time()},indent=2))
    print('ADE20K archive prepared',flush=True)


if __name__=='__main__':
    RUN.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures=[executor.submit(model),executor.submit(dataset)]
        for future in futures:future.result()
    print('DATA AND CHECKPOINT READY',flush=True)
