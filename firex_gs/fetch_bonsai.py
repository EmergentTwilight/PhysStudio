"""Fetch the released Graphdeco Bonsai model and cameras with ZIP CRC checks."""
import argparse, hashlib, json, urllib.request, zipfile
from pathlib import Path
from physstudio.paths import data_path
from rainy_gs.fetch_graphdeco import ModelZip, URL


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=data_path("firex_gs/graphdeco"))
    a = p.parse_args()
    with urllib.request.urlopen(
        urllib.request.Request(URL, headers={"Range": "bytes=0-0"}), timeout=60
    ) as r:
        if r.status != 206:
            raise IOError("Archive must support HTTP ranges")
        size = int(r.headers["Content-Range"].split("/")[-1])
    remote = ModelZip(URL, size)
    with zipfile.ZipFile(remote) as archive:
        infos = archive.infolist()
    selected = {
        f"bonsai/{name}"
        for name in [
            "cameras.json",
            "cfg_args",
            "point_cloud/iteration_30000/point_cloud.ply",
        ]
    }
    manifest = dict(
        source="https://github.com/graphdeco-inria/gaussian-splatting",
        url=URL,
        files={},
    )
    a.output.mkdir(parents=True, exist_ok=True)
    for info in infos:
        if info.filename not in selected:
            continue
        name, result = remote.extract_member(info, a.output)
        manifest["files"][name] = dict(
            bytes=info.file_size,
            crc32=info.CRC,
            sha256=hashlib.sha256((a.output / name).read_bytes()).hexdigest(),
        )
        (a.output / "bonsai_download.json").write_text(json.dumps(manifest, indent=2))
        print(name, result, flush=True)
    assert set(manifest["files"]) == selected


if __name__ == "__main__":
    main()
