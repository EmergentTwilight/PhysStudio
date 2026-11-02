"""Run, render, package, and inspect the three capillary-contact demos."""
import argparse
import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
from PIL import Image
from .scenes import DESCRIPTIONS

ROOT=Path(__file__).resolve().parents[1]
BLENDER=os.environ.get('BLENDER', 'blender')
FFMPEG=os.environ.get('FFMPEG', 'ffmpeg')


def encode(folder):
    config=json.loads((folder/'render_config.json').read_text())
    simulation=Path(config['input'])
    source=json.loads((simulation/'config.json').read_text())
    if not (simulation/'complete.json').exists():
        raise ValueError(f'Simulation is incomplete: {simulation}')
    if config['frames']!=list(range(source['frames'])):
        raise ValueError('Encoding requires every frame of the completed simulation.')
    frames=sorted((folder/'frames').glob('*.png'))
    assert len(frames)==len(config['frames'])
    assert config['frames']==list(range(len(frames)))
    records=json.loads((folder/'frames.json').read_text())
    assert [row['frame'] for row in records]==config['frames']
    for file in frames:
        with Image.open(file) as im:
            assert im.size==(config['width'],config['height']),file
    cmd=[str(FFMPEG),'-y','-hide_banner','-loglevel','warning','-framerate','30',
        '-i',str(folder/'frames/%05d.png'),'-an','-c:v','libx264','-preset','slow',
        '-crf','19','-pix_fmt','yuv420p','-threads','4','-movflags','+faststart',str(folder/'demo.mp4')]
    subprocess.run(cmd,check=True)
    with (folder/'decode.log').open('w') as log:
        subprocess.run([str(FFMPEG),'-v','error','-xerror','-i',str(folder/'demo.mp4'),
            '-f','null','-'],stdout=log,stderr=log,check=True)
    Image.open(frames[len(frames)//3]).save(folder/'poster.jpg',quality=92)
    return dict(file=str(folder/'demo.mp4'),frames=len(frames),fps=30,
                megabytes=round((folder/'demo.mp4').stat().st_size/1.e6,2),full_decode='passed')


def package(root, folders):
    """Export independent videos and looping previews for repository pages."""
    sources = []
    for folder in folders:
        render_config = json.loads((folder / 'render_config.json').read_text())
        simulation = Path(render_config['input'])
        config = json.loads((simulation / 'config.json').read_text())
        sources.append((config['scene'], folder, simulation, config))
    names = [row[0] for row in sources]
    if len(set(names))!=len(names) or any(name not in DESCRIPTIONS for name in names):
        raise ValueError('Provide complete render folders with unique scene names.')
    root.mkdir(parents=True, exist_ok=True)
    reports = []
    for name, folder, simulation, config in sources:
        report = encode(folder)
        for filename, extension in [('demo.mp4', '.mp4'), ('poster.jpg', '.jpg')]:
            shutil.copy2(folder / filename, root / (name + extension))
        gif = root / (name + '.gif')
        subprocess.run([
            str(FFMPEG), '-y', '-hide_banner', '-loglevel', 'warning',
            '-i', str(root / (name + '.mp4')), '-filter_complex_threads', '1',
            '-filter_complex',
            'fps=15,scale=480:-1:flags=lanczos,split[a][b];'
            '[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=sierra2_4a',
            '-loop', '0', str(gif)
        ], check=True)
        with Image.open(gif) as preview:
            gif_report = dict(width=preview.width, height=preview.height,
                              frames=preview.n_frames, loop=preview.info.get('loop'))
            for index in range(preview.n_frames):
                preview.seek(index)
                preview.load()
        assets = {}
        for extension in ['.mp4', '.gif', '.jpg']:
            path = root / (name + extension)
            assets[path.name] = dict(bytes=path.stat().st_size,
                                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        report.update(scene=name, source_render=str(folder.resolve()),
                      source_simulation=str(simulation.resolve()), assets=assets,
                      preview=gif_report, duration_seconds=report['frames']/report['fps'])
        reports.append(report)
        (root / 'media_manifest.json').write_text(json.dumps(reports, indent=2))
        print(json.dumps(report), flush=True)
    print('Independent media ready:', root.resolve(), flush=True)


def main():
    global FFMPEG
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    run=sub.add_parser('run');run.add_argument('--scene',choices=list(DESCRIPTIONS),required=True)
    run.add_argument('--output',type=Path,required=True);run.add_argument('--gpu',default='0')
    run.add_argument('--frames',type=int);run.add_argument('--resolution',type=int,default=96)
    run.add_argument('--blender',default=BLENDER);run.add_argument('--samples',type=int,default=32)
    run.add_argument('--width',type=int,default=1280)
    run.add_argument('--cpu',action='store_true',help='Render with Cycles on the CPU')
    run.add_argument('--stirrer',action='store_true')
    run.add_argument('--legacy-stirrer',action='store_true',help='Use the archived potential-only stirring model')
    run.add_argument('--dt',type=float);run.add_argument('--newton',type=int)
    run.add_argument('--settle',type=float)
    run.add_argument('--viscosity',type=float)
    run.add_argument('--pinned-rim',action='store_true')
    run.add_argument('--stage',choices=['all','simulate','render'],default='all')
    finish=sub.add_parser('package');finish.add_argument('--output',type=Path,required=True)
    finish.add_argument('--renders',type=Path,nargs='+',required=True)
    for command in [run, finish]:
        command.add_argument('--ffmpeg', default=FFMPEG)
    args=p.parse_args()
    if args.action=='run' and args.legacy_stirrer and not args.stirrer:
        p.error('--legacy-stirrer requires --stirrer')
    if args.action=='run' and (args.width<32 or args.width%2):
        p.error('--width must be even and at least 32')
    if args.action=='run' and args.frames is not None and args.frames<1:
        p.error('--frames must be positive')
    FFMPEG=args.ffmpeg
    if args.action=='package':return package(args.output,args.renders)
    if args.frames is None:args.frames=240 if args.stirrer else 180
    if args.dt is None:args.dt=1/480 if args.stirrer else 1/360
    if args.newton is None:args.newton=10 if args.stirrer and not args.legacy_stirrer else (8 if args.stirrer else 6)
    if args.settle is None:args.settle=.18 if args.stirrer else 0.
    if args.viscosity is None:args.viscosity=.008 if args.legacy_stirrer and args.scene=='capillary_cluster' else .003
    args.output=args.output.expanduser().resolve()
    sim=args.output/'simulation';render=args.output/'render'
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',CUDA_VISIBLE_DEVICES=args.gpu,
        XDG_CACHE_HOME=str(ROOT/'.cache/blender'),CUDA_CACHE_PATH=str(ROOT/'.cache/cuda'),
        OPTIX_CACHE_PATH=str(ROOT/'.cache/optix'))
    if args.stage in ['all','simulate']:
        command=[sys.executable,'-m','surface_tension.simulate','--scene',args.scene,
            '--output',str(sim),'--frames',str(args.frames),'--resolution',str(args.resolution),
            '--dt',str(args.dt),'--frame-dt',str(1/120),'--newton',str(args.newton),
            '--settle',str(args.settle),'--viscosity',str(args.viscosity)]
        if args.stirrer:
            command.append('--stirrer')
            if not args.legacy_stirrer:command.append('--surface-flow')
        if args.pinned_rim or (args.stirrer and args.scene=='capillary_cluster'):
            command.append('--pinned-rim')
        subprocess.run(command,env=env,check=True)
        if args.stirrer and not args.legacy_stirrer:
            subprocess.run([sys.executable,'-m','surface_tension.flow_checks','--output',
                str(args.output/'flow_validation.json'),'--simulations',str(sim)],env=env,check=True)
        subprocess.run([sys.executable,'-m','surface_tension.validate','--output',
            str(args.output/'validation.json'),'--simulations',str(sim)],env=env,check=True)
    if args.stage in ['all','render']:
        command=[str(args.blender),'-b','--factory-startup','--python',
            str(ROOT/'surface_tension/render_blender.py'),'--','--input',str(sim),
            '--output',str(render),'--samples',str(args.samples),
            '--width',str(args.width),'--height',str(round(args.width*5/16)*2)]
        if args.cpu:command.append('--cpu')
        subprocess.run(command,env=env,check=True)
        print(json.dumps(encode(render),indent=2))


if __name__=='__main__':main()
