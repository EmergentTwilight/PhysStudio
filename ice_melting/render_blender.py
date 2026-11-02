"""Cycles rendering of cached ice and meltwater surfaces."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'.cache/blender'))
os.environ.setdefault('CUDA_CACHE_PATH',str(ROOT/'.cache/cuda'))
os.environ.setdefault('OPTIX_CACHE_PATH',str(ROOT/'.cache/optix'))
import bpy
import numpy as np
from mathutils import Vector

SCALE=20.


def material(name,color,rough=.3,metal=0.):
    mat=bpy.data.materials.new(name);mat.use_nodes=True
    p=mat.node_tree.nodes.get('Principled BSDF');p.inputs['Base Color'].default_value=(*color,1)
    p.inputs['Roughness'].default_value=rough;p.inputs['Metallic'].default_value=metal
    return mat


def transparent(name,ior,color,rough,absorption,scatter):
    mat=material(name,(.97,.99,1.),rough)
    nt=mat.node_tree;p=nt.nodes.get('Principled BSDF');p.inputs['Transmission Weight'].default_value=1.
    p.inputs['IOR'].default_value=ior
    p.inputs['Coat Weight'].default_value=.08
    absorb=nt.nodes.new('ShaderNodeVolumeAbsorption');absorb.inputs['Color'].default_value=(*color,1)
    absorb.inputs['Density'].default_value=absorption
    if scatter:
        scat=nt.nodes.new('ShaderNodeVolumeScatter');scat.inputs['Color'].default_value=(.85,.92,.95,1)
        noise=nt.nodes.new('ShaderNodeTexNoise');noise.inputs['Scale'].default_value=5.
        ramp=nt.nodes.new('ShaderNodeValToRGB')
        ramp.color_ramp.elements[0].position=.35;ramp.color_ramp.elements[0].color=(.005,.005,.005,1)
        ramp.color_ramp.elements[1].position=.75;ramp.color_ramp.elements[1].color=(scatter,scatter,scatter,1)
        nt.links.new(noise.outputs['Fac'],ramp.inputs[0]);nt.links.new(ramp.outputs['Color'],scat.inputs['Density'])
        scat.inputs['Anisotropy'].default_value=.15
        add=nt.nodes.new('ShaderNodeAddShader');nt.links.new(absorb.outputs[0],add.inputs[0]);nt.links.new(scat.outputs[0],add.inputs[1])
        nt.links.new(add.outputs[0],nt.nodes.get('Material Output').inputs['Volume'])
        bump=nt.nodes.new('ShaderNodeBump');bump.inputs['Strength'].default_value=.09;bump.inputs['Distance'].default_value=.012
        detail=nt.nodes.new('ShaderNodeTexNoise');detail.inputs['Scale'].default_value=35.;detail.inputs['Detail'].default_value=3.
        nt.links.new(detail.outputs['Fac'],bump.inputs['Height']);nt.links.new(bump.outputs['Normal'],p.inputs['Normal'])
    else:nt.links.new(absorb.outputs[0],nt.nodes.get('Material Output').inputs['Volume'])
    return mat


def cube(name,location,dimensions,mat,bevel=.02):
    bpy.ops.mesh.primitive_cube_add(size=1,location=location)
    obj=bpy.context.object;obj.name=name;obj.dimensions=dimensions
    bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    obj.data.materials.append(mat)
    if bevel:
        mod=obj.modifiers.new('Rounded edges','BEVEL');mod.width=bevel;mod.segments=4
        obj.modifiers.new('Corner normals','WEIGHTED_NORMAL')
    return obj


def cylinder(name,center,radius,depth,mat):
    bpy.ops.mesh.primitive_cylinder_add(vertices=128,radius=radius,depth=depth,location=center)
    obj=bpy.context.object;obj.name=name;obj.data.materials.append(mat)
    mod=obj.modifiers.new('Rim finish','BEVEL');mod.width=.018;mod.segments=3
    obj.modifiers.new('Weighted normals','WEIGHTED_NORMAL');return obj


def area(name,location,target,power,size,size_y,color=(1,1,1)):
    lamp=bpy.data.lights.new(name,'AREA');lamp.energy=power;lamp.shape='RECTANGLE';lamp.size=size;lamp.size_y=size_y;lamp.color=color
    obj=bpy.data.objects.new(name,lamp);bpy.context.collection.objects.link(obj);obj.location=location
    obj.rotation_euler=(Vector(target)-obj.location).to_track_quat('-Z','Y').to_euler()


def dynamic(name,mat):
    mesh=bpy.data.meshes.new(name);obj=bpy.data.objects.new(name,mesh);bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat)
    smooth=obj.modifiers.new('Surface regularity','SMOOTH');smooth.factor=.6;smooth.iterations=3
    return obj


def set_mesh(obj,vertices,faces):
    obj.hide_render=len(faces)==0
    mesh=obj.data;mesh.clear_geometry()
    if len(faces):
        mesh.from_pydata((vertices*SCALE).tolist(),[],faces.tolist());mesh.update()
        for polygon in mesh.polygons:polygon.use_smooth=True


def setup(cfg,args):
    bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
    scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.samples=args.samples
    scene.cycles.use_denoising=True;scene.cycles.denoiser='OPENIMAGEDENOISE';scene.cycles.denoising_use_gpu=not args.cpu
    scene.cycles.max_bounces=12;scene.cycles.transmission_bounces=10;scene.cycles.transparent_max_bounces=10
    scene.cycles.glossy_bounces=6;scene.cycles.volume_bounces=2;scene.cycles.adaptive_threshold=.02
    scene.cycles.seed=19;scene.cycles.use_animated_seed=False
    if not args.cpu:
        preferences=bpy.context.preferences.addons['cycles'].preferences;preferences.compute_device_type='OPTIX'
        preferences.get_devices()
        for device in preferences.devices:device.use=device.type=='OPTIX'
        scene.cycles.device='GPU';print('Render devices',[d.name for d in preferences.devices if d.use],flush=True)
    scene.render.resolution_x=args.width;scene.render.resolution_y=args.height;scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB'
    scene.view_settings.view_transform='AgX';scene.view_settings.look='AgX - Medium High Contrast';scene.view_settings.exposure=.25
    world=bpy.data.worlds.new('Neutral studio');scene.world=world;world.use_nodes=True
    world.node_tree.nodes['Background'].inputs['Color'].default_value=(.19,.20,.22,1)
    world.node_tree.nodes['Background'].inputs['Strength'].default_value=.35
    black=material('Graphite ceramic',(.045,.052,.06),.28)
    white=material('Porcelain',(.42,.46,.49),.30)
    metal=material('Brushed stainless',(.38,.40,.42),.26,.85)
    copper=material('Brushed copper',(.43,.19,.085),.28,.78)
    glass=transparent('Borosilicate glass',1.46,(.96,.985,.99),.035,.05,0.)
    water=transparent('Meltwater',1.333,(.88,.965,.985),.025,.12,0.)
    ice=transparent('Glacial ice',1.309,(.76,.89,.96),.09,.30,1.8 if cfg['scene']=='cubes' else .45)
    cube('Studio floor',(0,0,-.24),(200,200,.15),material('Warm neutral stone',(.14,.15,.16),.65),0.)
    lo,hi=np.array(cfg['bounds'])*SCALE;length,width=hi[:2]-lo[:2]
    if cfg['scene']=='bunny':
        base=copper;rim=black
        cube('Copper heating plate',(0,0,-.06),(length+.12,width+.12,.12),base,.06)
        # Fine plate grooves provide reflected and refracted geometry cues.
        for y in np.linspace(lo[1]+.12,hi[1]-.12,19):
            cube('Engraved groove',(0,y,.0005),(length-.10,.008,.001),black,0.)
        cylinder('Heat control',(hi[0]+.16,0.,-.045),.12,.08,metal)
    elif cfg['scene']=='arch':
        base=white;rim=black
        cube('Porcelain collection tray',(0,0,-.06),(length+.14,width+.14,.12),base,.07)
        for x in np.linspace(lo[0]+.1,hi[0]-.1,19):
            cube('Tray inlay',(x,0,.0005),(.008,width-.1,.001),material('Slate inlay',(.24,.28,.30),.3),0.)
    else:
        base=white;rim=glass
        cube('Bath floor',(0,0,-.07),(length+.10,width+.10,.14),material('Dark slate bath',(.055,.075,.085),.32),.035)
        for x in np.arange(lo[0]+.1,hi[0],.25):
            for y in np.arange(lo[1]+.1,hi[1],.25):
                cylinder('Basin dot',(x,y,.0005),.014,.001,metal)
    rimheight=.020*SCALE if cfg['scene']!='cubes' else .045*SCALE
    for sign in [-1,1]:
        cube('Long rim',(0,sign*(width/2+.025),rimheight/2-.03),(length+.16,.05,rimheight),rim,.018)
        cube('Short rim',(sign*(length/2+.025),0,rimheight/2-.03),(.05,width+.10,rimheight),rim,.018)
    area('Large softbox',(-3.5,-4.0,7.0),(0,0,.5),1500,5.,3.)
    area('Ice rim',(3.8,2.5,5.0),(0,0,1.),1800,2.8,4.5,(.88,.94,1.))
    area('Fill',(-4.5,1.5,3.5),(0,0,.7),900,3.0,4.)
    area('Long reflection',(1.0,-4.0,3.5),(0,0,.3),450,4.,.4)
    iceobj=dynamic('Ice surface',ice);waterobj=dynamic('Water surface',water)
    bpy.ops.object.camera_add();camera=bpy.context.object;camera.name='Orbit camera';scene.camera=camera
    camera.data.lens=54;camera.data.clip_start=.01;camera.data.clip_end=300
    camera.data.dof.use_dof=False
    return scene,iceobj,waterobj,camera


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--meshes',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--width',type=int,default=1280);p.add_argument('--height',type=int,default=800)
    p.add_argument('--samples',type=int,default=48);p.add_argument('--cpu',action='store_true')
    p.add_argument('--frames',type=int,nargs='*');p.add_argument('--resume',action='store_true')
    args=p.parse_args(sys.argv[sys.argv.index('--')+1:]);args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'frames').mkdir(exist_ok=True)
    cfg=json.loads((args.input/'config.json').read_text())
    indices=args.frames if args.frames is not None else list(range(cfg['frames']))
    if not indices or any(i<0 or i>=cfg['frames'] for i in indices):
        raise ValueError('Requested frames are outside the cached simulation')
    missing=[i for i in indices if not (args.meshes/f'{i:05d}.npz').is_file()]
    if missing:
        raise ValueError(f'Surface cache is incomplete; first missing frame: {missing[0]}')
    scene,ice,water,camera=setup(cfg,args)
    settings=dict(input=str(args.input.resolve()),meshes=str(args.meshes.resolve()),frames=indices,
                  width=args.width,height=args.height,samples=args.samples)
    previous=args.output/'render_config.json'
    if args.resume and previous.exists() and json.loads(previous.read_text())!=settings:
        raise ValueError('Resume requires matching render settings')
    previous.write_text(json.dumps(settings,indent=2))
    records=[];start=time.time()
    for index in indices:
        mesh=np.load(args.meshes/f'{index:05d}.npz')
        set_mesh(ice,mesh['ice_vertices'],mesh['ice_faces']);set_mesh(water,mesh['water_vertices'],mesh['water_faces'])
        t=index/(cfg.get('camera_frames',cfg['frames'])-1);angle=math.radians(-105+78*t)
        radius=7.8 if cfg['scene']!='arch' else 8.2
        elevation=4.7 if cfg['scene']!='cubes' else 5.8
        camera.location=(radius*math.cos(angle),radius*math.sin(angle),elevation+.35*math.sin(math.pi*t))
        target=Vector(np.array(cfg['target'])*SCALE)
        camera.rotation_euler=(target-camera.location).to_track_quat('-Z','Y').to_euler()
        scene.frame_set(index+1);file=(args.output/'frames'/f'{index:05d}.png').resolve()
        if not (args.resume and file.exists()):
            scene.render.filepath=str(file);bpy.ops.render.render(write_still=True)
        records.append(dict(frame=index,camera=list(camera.location),elapsed=time.time()-start))
        (args.output/'frames.json').write_text(json.dumps(records,indent=2))
        print(json.dumps(records[-1]),flush=True)


if __name__=='__main__':main()
