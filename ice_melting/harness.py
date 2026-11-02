"""Run, validate, render, and package the ice-melting demos."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from PIL import Image,ImageDraw,ImageFont
from .scenes import PRESETS

ROOT=Path(__file__).resolve().parents[1]
BLENDER=os.environ.get('BLENDER','blender')
FFMPEG=os.environ.get('FFMPEG','ffmpeg')


def encode(folder):
    r=json.loads((folder/'render_config.json').read_text())
    cfg=json.loads((Path(r['input'])/'config.json').read_text())
    frames=sorted((folder/'frames').glob('*.png'))
    assert r['frames']==list(range(cfg['frames']))
    assert len(frames)==cfg['frames']
    assert [f.stem for f in frames]==[f'{i:05d}' for i in r['frames']]
    for file in frames:
        with Image.open(file) as image:
            assert image.size==(r['width'],r['height'])
            image.verify()
    out=folder/'demo.mp4'
    subprocess.run([FFMPEG,'-y','-v','error','-framerate','30','-i',str(folder/'frames/%05d.png'),
        '-an','-c:v','libx264','-preset','slow','-crf','19','-pix_fmt','yuv420p','-threads','4',
        '-movflags','+faststart',str(out)],check=True)
    subprocess.run([FFMPEG,'-v','error','-xerror','-i',str(out),'-f','null','-'],check=True)
    return cfg,out


def package(output,renders):
    output.mkdir(parents=True,exist_ok=False)
    manifest=[];tiles=[]
    for folder in renders:
        cfg,video=encode(folder);name=cfg['scene']
        assert name not in [r['scene'] for r in manifest]
        target=output/f'{name}.mp4';shutil.copy2(video,target)
        gif=output/f'{name}.gif'
        subprocess.run([FFMPEG,'-v','error','-y','-i',str(target),'-filter_complex_threads','1',
            '-filter_complex','fps=15,scale=480:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse=dither=sierra2_4a',
            '-loop','0',str(gif)],check=True)
        idx=[0,round((cfg['frames']-1)/3),round((cfg['frames']-1)*2/3),cfg['frames']-1]
        row=Image.new('RGB',(4*400,290),'#f2f2f2');draw=ImageDraw.Draw(row)
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
        for column,index in enumerate(idx):
            image=Image.open(folder/'frames'/f'{index:05d}.png').convert('RGB');image.thumbnail((400,250))
            row.paste(image,(column*400,32));draw.text((column*400+12,7),f"{cfg['label']}  |  {index/30:.1f} s",font=font,fill='#252525')
        tiles.append(row)
        Image.open(folder/'frames'/f'{idx[1]:05d}.png').save(output/f'{name}.jpg',quality=92)
        manifest.append(dict(scene=name,label=cfg['label'],input=str(folder.resolve()),
            video=target.name,frames=cfg['frames'],fps=30,width=Image.open(folder/'frames/00000.png').width,
            physical_seconds=cfg['seconds'],thermal_scale=cfg['thermal_scale'],
            sha256=hashlib.sha256(target.read_bytes()).hexdigest(),bytes=target.stat().st_size))
    sheet=Image.new('RGB',(1600,290*len(tiles)),'white')
    for i,tile in enumerate(tiles):sheet.paste(tile,(0,290*i))
    sheet.save(output/'contact_sheet.jpg',quality=94)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Ice Melting</title><style>body{background:#eceeef;color:#20262c;font:16px/1.6 system-ui;margin:0}main{max-width:1400px;margin:auto;padding:40px 24px}h1{font-size:48px;letter-spacing:-2px}section{margin:32px 0}video{width:100%;border-radius:8px}a{color:#285772}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:22px}@media(max-width:900px){.grid{grid-template-columns:1fr}}</style><main><h1>Ice, heat &amp; flowing water.</h1><p>Particle enthalpy, rigid ice fragments, and refractive meltwater.</p><div class="grid">'''
    for item in manifest:
        html+=f'<section><h2>{item["label"]}</h2><video src="{item["video"]}" poster="{item["scene"]}.jpg" controls autoplay muted loop playsinline></video></section>'
    html+='</div><p><a href="contact_sheet.jpg">Melting progression</a></p></main></html>'
    (output/'index.html').write_text(html)
    print(json.dumps(manifest,indent=2),flush=True)


def main():
    global FFMPEG
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    run=sub.add_parser('run');run.add_argument('--scene',choices=list(PRESETS),required=True)
    run.add_argument('--output',type=Path,required=True);run.add_argument('--gpu',default='0')
    run.add_argument('--frames',type=int,default=240);run.add_argument('--seconds',type=float)
    run.add_argument('--dx',type=float);run.add_argument('--thermal-scale',type=float)
    run.add_argument('--settings',type=Path,help='JSON wetting and surface-reconstruction overrides')
    run.add_argument('--width',type=int,default=1280);run.add_argument('--samples',type=int,default=48)
    run.add_argument('--blender',type=Path,default=BLENDER)
    run.add_argument('--cpu',action='store_true',help='Render with Cycles on the CPU')
    run.add_argument('--stage',choices=['all','simulate','mesh','render'],default='all')
    pack=sub.add_parser('package');pack.add_argument('--output',type=Path,required=True)
    pack.add_argument('--renders',type=Path,nargs='+',required=True)
    for command in [run,pack]:command.add_argument('--ffmpeg',default=FFMPEG)
    args=p.parse_args()
    FFMPEG=args.ffmpeg
    if args.action=='package':return package(args.output,args.renders)
    if args.frames<2:p.error('--frames must be at least two')
    if args.width<32 or args.width%2:p.error('--width must be even and at least 32')
    if args.seconds is None:args.seconds=4.*(args.frames-1)/239.
    if args.seconds<=0.:p.error('--seconds must be positive')
    args.output=args.output.expanduser().resolve()
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=args.gpu,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    for key,folder in [('WARP_CACHE_PATH','warp'),('XDG_CACHE_HOME','blender'),
                       ('CUDA_CACHE_PATH','cuda'),('OPTIX_CACHE_PATH','optix')]:
        env.setdefault(key,str(ROOT/'.cache'/folder))
    sim=args.output/'simulation';mesh=args.output/'meshes';render=args.output/'render'
    if args.stage in ['all','simulate']:
        cmd=[sys.executable,'-m','ice_melting.simulate','--scene',args.scene,'--output',str(sim),
             '--frames',str(args.frames),'--seconds',str(args.seconds)]
        if args.dx is not None:cmd+=['--dx',str(args.dx)]
        if args.thermal_scale is not None:cmd+=['--thermal-scale',str(args.thermal_scale)]
        if args.settings is not None:cmd+=['--settings',str(args.settings)]
        subprocess.run(cmd,env=env,check=True)
        subprocess.run([sys.executable,'-m','ice_melting.validate','--simulations',str(sim),
            '--output',str(args.output/'validation.json')],env=env,check=True)
    if args.stage in ['all','mesh']:
        subprocess.run([sys.executable,'-m','ice_melting.surface','--input',str(sim),'--output',str(mesh)],env=env,check=True)
        subprocess.run([sys.executable,'-m','ice_melting.validate','--skip-numerics','--meshes',str(mesh),
            '--output',str(args.output/'mesh_validation.json')],env=env,check=True)
    if args.stage in ['all','render']:
        command=[str(args.blender),'-b','--factory-startup','--threads','8','--python',str(Path(__file__).with_name('render_blender.py')),
            '--','--input',str(sim),'--meshes',str(mesh),'--output',str(render),'--samples',str(args.samples),
            '--width',str(args.width),'--height',str(round(args.width*5/16)*2)]
        if args.cpu:command.append('--cpu')
        subprocess.run(command,env=env,check=True)
        encode(render)


if __name__=='__main__':main()
