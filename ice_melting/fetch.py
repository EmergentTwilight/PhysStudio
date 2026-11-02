"""Download and verify the Stanford Bunny mesh."""
import hashlib
import json
from urllib.request import Request, urlopen
from physstudio.paths import data_path

BUNNY='https://github.com/isl-org/open3d_downloads/releases/download/20220201-data/BunnyMesh.ply'
BUNNY_SHA256='b1acc63bece78444aa2e15bdcc72371a201279b98c6f5d4b74c993d02f0566fe'


def main():
    out=data_path('ice_melting');out.mkdir(parents=True,exist_ok=True)
    path=out/'bunny.ply'
    if not path.exists():
        request=Request(BUNNY,headers={'User-Agent':'PhysStudio/0.1'})
        with urlopen(request,timeout=90) as response:content=response.read()
        if hashlib.sha256(content).hexdigest()!=BUNNY_SHA256:
            raise ValueError('Bunny download checksum differs from the pinned Open3D release')
        path.write_bytes(content)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=BUNNY_SHA256:
        raise ValueError('Bunny asset checksum differs from the pinned Open3D release')
    record=dict(asset='Stanford Bunny',source='https://graphics.stanford.edu/data/3Dscanrep/',
        download=BUNNY,mirror='Open3D public data release 20220201',sha256=digest)
    (out/'source.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record),flush=True)


if __name__=='__main__':main()
