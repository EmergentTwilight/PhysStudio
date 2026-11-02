"""Download the CC0 Poly Haven orchard models and record their checksums."""
import hashlib
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ['food_apple_01', 'dry_quiver_leaf', 'wooden_bowl_02']


def download(url, timeout=120):
    request = Request(url, headers={'User-Agent': 'PhysStudio/0.1'})
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def main():
    root = Path(os.environ.get('PHYSSTUDIO_DATA_ROOT', str(ROOT / 'data'))).expanduser().resolve()
    root = root / 'surface_tension'
    records = []
    for asset in ASSETS:
        folder = root / asset
        folder.mkdir(parents=True, exist_ok=True)
        metadata = json.loads(download(f'https://api.polyhaven.com/files/{asset}', timeout=60))
        spec = metadata['gltf']['1k']['gltf']
        files = dict(spec.get('include', {}))
        files[f'{asset}_1k.gltf'] = spec
        hashes = {}
        for name, item in files.items():
            path = (folder / name).resolve()
            if not path.is_relative_to(folder):
                raise ValueError(f'Asset path leaves its download directory: {name}')
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists() or hashlib.md5(path.read_bytes()).hexdigest() != item['md5']:
                content = download(item['url'])
                if hashlib.md5(content).hexdigest() != item['md5']:
                    raise ValueError(f'Download checksum mismatch: {name}')
                path.write_bytes(content)
            hashes[name] = dict(url=item['url'], bytes=path.stat().st_size,
                               sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        record = dict(asset=asset, source=f'https://polyhaven.com/a/{asset}',
                      license='CC0-1.0', files=hashes)
        (folder / 'source.json').write_text(json.dumps(record, indent=2))
        records.append(record)
        print('Downloaded and verified', asset, flush=True)
    (root / 'sources.json').write_text(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
