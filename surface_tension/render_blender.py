"""Cycles macro photography of cached capillary-contact states."""
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
from mathutils import Vector, Matrix

ASSET_ROOT=Path(os.environ.get('PHYSSTUDIO_DATA_ROOT',str(ROOT/'data'))).expanduser().resolve()
SCALE=100.


def material(name,color,metal=0.,rough=.3,transmission=0.,ior=1.45):
    mat=bpy.data.materials.new(name);mat.use_nodes=True
    bsdf=mat.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value=(*color,1)
    bsdf.inputs['Metallic'].default_value=metal
    bsdf.inputs['Roughness'].default_value=rough
    bsdf.inputs['Transmission Weight'].default_value=transmission
    bsdf.inputs['IOR'].default_value=ior
    return mat


def cube(name,position,size,mat,bevel=0.):
    bpy.ops.mesh.primitive_cube_add(size=1,location=position)
    obj=bpy.context.object;obj.name=name;obj.dimensions=size
    bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    obj.data.materials.append(mat)
    if bevel:
        mod=obj.modifiers.new('Soft machined edges','BEVEL');mod.width=bevel;mod.segments=5
        mod=obj.modifiers.new('Weighted normals','WEIGHTED_NORMAL')
    return obj


def sphere(name,position,scale,mat):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48,ring_count=24,location=position)
    obj=bpy.context.object;obj.name=name;obj.scale=scale;obj.data.materials.append(mat)
    for face in obj.data.polygons:face.use_smooth=True
    return obj


def curve(name,points,radius,mat):
    data=bpy.data.curves.new(name,'CURVE');data.dimensions='3D';data.resolution_u=2
    data.bevel_depth=radius;data.bevel_resolution=5
    spline=data.splines.new('POLY');spline.points.add(len(points)-1)
    for vert,point in zip(spline.points,points):vert.co=(*point,1)
    obj=bpy.data.objects.new(name,data);bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat)
    return obj


def imported_asset(asset):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.import_scene.gltf(filepath=str(ASSET_ROOT/f'surface_tension/{asset}/{asset}_1k.gltf'))
    objects=[o for o in bpy.context.selected_objects if o.type=='MESH']
    for obj in objects:
        obj.data.transform(obj.matrix_world)
        obj.matrix_world=Matrix.Identity(4)
    return objects


def light(name,location,target,power,size,shape='DISK',size_y=None,color=(1,1,1)):
    data=bpy.data.lights.new(name,'AREA');data.energy=power;data.shape=shape;data.size=size
    if size_y is not None:data.size_y=size_y
    data.color=color
    obj=bpy.data.objects.new(name,data);bpy.context.collection.objects.link(obj)
    obj.location=location;obj.rotation_euler=(Vector(target)-obj.location).to_track_quat('-Z','Y').to_euler()


def setup(config,args,mesh):
    bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
    scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.samples=args.samples
    scene.cycles.use_denoising=True;scene.cycles.denoiser=args.denoiser
    scene.cycles.denoising_use_gpu=not args.cpu
    scene.cycles.max_bounces=10
    scene.cycles.transmission_bounces=8;scene.cycles.glossy_bounces=5
    scene.cycles.transparent_max_bounces=8
    scene.cycles.adaptive_threshold=.025
    scene.render.resolution_x=args.width;scene.render.resolution_y=args.height
    scene.render.resolution_percentage=100;scene.render.image_settings.file_format='PNG'
    scene.render.image_settings.color_mode='RGB';scene.render.image_settings.color_depth='8'
    scene.render.film_transparent=False
    scene.render.threads_mode='FIXED';scene.render.threads=6
    scene.render.use_persistent_data=True
    if not args.cpu:
        prefs=bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type='OPTIX';prefs.get_devices()
        found=[]
        for device in prefs.devices:
            device.use=device.type=='OPTIX'
            if device.use:found.append(device.name)
        if not found:raise RuntimeError('No OptiX device visible')
        scene.cycles.device='GPU'
        print('Render devices',found,flush=True)
    scene.view_settings.view_transform='AgX'
    scene.view_settings.look='AgX - Medium High Contrast'
    scene.view_settings.exposure=.15
    scene.world.use_nodes=True
    world=scene.world.node_tree.nodes.get('Background');world.inputs[0].default_value=(.16,.20,.24,1)
    world.inputs[1].default_value=.55
    length,width,depth=config['length']*SCALE,config['width']*SCALE,config['depth']*SCALE
    # The close rectangular basin exposes the actual rectangular simulation domain.
    ceramic=material('Deep graphite ceramic',(.025,.044,.049),.18,.22)
    base=material('Slate beneath the water',(.18,.22,.24),.05,.39)
    edge=material('Brushed nickel',(.31,.37,.38),.88,.24)
    if config['scene']=='capillary_cluster':
        ceramic=material('Porcelain',(.72,.74,.72),.0,.22)
        base=material('Sea glass glaze',(.17,.32,.34),.0,.30)
    if config['scene']=='orchard':
        ceramic=material('Jade glaze',(.035,.12,.09),.12,.21)
        base=material('Jade basin bed',(.16,.24,.17),.02,.29)
    zbed=-depth
    cube('Basin foundation',(0,0,zbed-.14),(length+.32,width+.32,.20),ceramic,.10)
    cube('Basin bed',(0,0,zbed-.025),(length,width,.05),base,.02)
    for x in [-1,1]:
        cube('Side rim',(x*(length/2+.07),0,-depth/2+.035),(.14,width+.29,depth+.20),ceramic,.055)
    for y in [-1,1]:
        cube('End rim',(0,y*(width/2+.07),-depth/2+.035),(length+.14,.14,depth+.20),ceramic,.055)
    ground=material('Neutral photographic backdrop',(.105,.12,.13),0,.43)
    cube('Table',(0,0,zbed-.26),(200,200,.16),ground,.02)
    # Fine engraved joints make the physically refracted image legible.
    line_mat=material('Submerged fine grout',(.04,.065,.07),0,.48)
    for x in np.arange(-length/2+.20,length/2,.38):
        cube('Fine submerged joint',(float(x),0,zbed+.002),(.006,width,.003),line_mat)
    for y in np.arange(-width/2+.17,width/2,.38):
        cube('Fine submerged joint',(0,float(y),zbed+.002),(length,.006,.003),line_mat)
    water=material('Water',(.97,.99,1.),0,.042 if config.get('surface_flow') else .026,1,1.333)
    absorption=water.node_tree.nodes.new('ShaderNodeVolumeAbsorption')
    absorption.inputs['Color'].default_value=(.45,.80,.83,1)
    absorption.inputs['Density'].default_value=.12
    output=water.node_tree.nodes.get('Material Output')
    water.node_tree.links.new(absorption.outputs['Volume'],output.inputs['Volume'])
    xy=mesh['xy']*SCALE
    vertices=np.column_stack([xy,np.zeros(len(xy))])
    nx,ny=config['nx'],config['ny']
    ids=np.arange(nx*ny).reshape(nx,ny)
    perimeter=np.concatenate([ids[:,0],ids[-1,1:],ids[-2::-1,-1],ids[0,-2:0:-1]])
    bottom=np.column_stack([xy[perimeter],np.full(len(perimeter),zbed+.008)])
    verts=np.concatenate([vertices,bottom])
    faces=mesh['faces'].tolist();n=len(vertices);nb=len(perimeter)
    for k in range(nb):
        kn=(k+1)%nb
        faces.append([int(perimeter[k]),n+k,n+kn,int(perimeter[kn])])
    faces.append(list(range(n+nb-1,n-1,-1)))
    data=bpy.data.meshes.new('Simulated liquid membrane');data.from_pydata(verts.tolist(),[],faces);data.update()
    water_obj=bpy.data.objects.new('Water volume',data);bpy.context.collection.objects.link(water_obj)
    water_obj.data.materials.append(water)
    for i,poly in enumerate(data.polygons):poly.use_smooth=i<len(mesh['faces'])
    metal=material('Polished steel',(.46,.51,.53),1.,.16)
    roots=[]
    for i,visual in enumerate(config['visual']):
        root=bpy.data.objects.new(f'Rigid body {i}',None);bpy.context.collection.objects.link(root)
        roots.append(root)
        if visual['kind']=='paperclip':
            mat=material(f'Metal {i}',visual['color'],1.,.18)
            points=np.asarray(visual['path'])*SCALE
            obj=curve('Solid wire',points,visual['radius']*SCALE,mat);obj.parent=root
            for p in [points[0],points[-1]]:
                sphere('Rounded wire end',p,[visual['radius']*SCALE]*3,mat).parent=root
        elif visual['kind']=='pin':
            axes=np.asarray(visual['axes'])*SCALE
            mat=material(f'Enamel {i}',visual['color'],.28,.18)
            sphere('Floating metal disc',(0,0,0),axes,metal).parent=root
            obj=sphere('Enamel handle',(0,0,.19),(.091,.091,.12),mat);obj.parent=root
            bpy.ops.mesh.primitive_cylinder_add(vertices=32,radius=.025,depth=.18,location=(0,0,.09))
            bpy.context.object.data.materials.append(metal);bpy.context.object.parent=root
        else:
            parts=imported_asset('food_apple_01')
            points=np.concatenate([np.asarray([v.co[:] for v in obj.data.vertices]) for obj in parts])
            lo,hi=np.quantile(points,[.002,.992],axis=0);center=(lo+hi)/2
            axes=np.asarray(visual['axes'])*SCALE
            factor=axes/((hi-lo)*.5)
            for obj in parts:
                for vert in obj.data.vertices:vert.co=(np.asarray(vert.co)-center)*factor
                obj.parent=root
                for poly in obj.data.polygons:poly.use_smooth=True
    if config['scene']=='orchard':
        # New scanned props remain outside the simulated basin.
        for asset,location,target_size,angle in [
            ('wooden_bowl_02',(3.4,2.8,zbed-.15),2.1,.25),
            ('dry_quiver_leaf',(-3.15,-.25,zbed-.12),1.6,-.65)]:
            parts=imported_asset(asset)
            points=np.concatenate([np.asarray([v.co[:] for v in o.data.vertices]) for o in parts])
            lo,hi=points.min(0),points.max(0);center=(lo+hi)/2;center[2]=lo[2]
            factor=target_size/max(hi-lo)
            for obj in parts:
                for v in obj.data.vertices:v.co=(np.asarray(v.co)-center)*factor
                obj.rotation_euler[2]=angle;obj.location=location
    probe_root=None
    if config.get('stirrer'):
        probe=config['stirrer'];radius=probe['radius']*SCALE
        length=probe['length']*SCALE
        probe_root=bpy.data.objects.new('Driven stirrer',None)
        bpy.context.collection.objects.link(probe_root)
        shaft=material('Brushed stirrer steel',(.34,.39,.42),.92,.24)
        sphere('Rounded stirring tip',(0,0,0),(radius,)*3,shaft).parent=probe_root
        bpy.ops.mesh.primitive_cylinder_add(vertices=64,radius=radius,depth=length,
                                            location=(0,0,length/2))
        obj=bpy.context.object;obj.name='Stirring shaft';obj.data.materials.append(shaft)
        obj.parent=probe_root
        for face in obj.data.polygons:face.use_smooth=True
        grip=material('Graphite stirrer grip',(.035,.045,.055),.15,.36)
        bpy.ops.mesh.primitive_cylinder_add(vertices=64,radius=radius*1.45,
                                            depth=.65,location=(0,0,length-.10))
        obj=bpy.context.object;obj.name='Graphite handle';obj.data.materials.append(grip)
        obj.parent=probe_root
        for face in obj.data.polygons:face.use_smooth=True
    if config.get('surface_flow'):
        light('Soft overhead',(-3,-1,6),(0,0,0),180,5.)
        light('Wide rim',(1,3,4),(0,0,-.1),55,4.,'RECTANGLE',1.7,(.90,.96,1.))
        light('Side accent',(-3,1,2.8),(0,0,0),18,3.,'RECTANGLE',.8)
        light('Front fill',(2,-4,2.2),(0,0,-.2),65,4.)
    else:
        light('Large softbox',(-3,-1,5),(0,0,0),220,2.2 if probe_root else 4.5)
        light('Rim strip',(1,3,3.5),(0,0,-.1),100,3.5,'RECTANGLE',.30,(.82,.92,1.))
        light('Long reflection',(-2.5,2,2.4),(0,0,0),45,3.,'RECTANGLE',.18)
        light('Front fill',(2,-4,1.8),(0,0,-.2),80,3.)
    bpy.ops.object.camera_add(location=(4,-6,4))
    camera=bpy.context.object;camera.name='Macro orbit';scene.camera=camera
    camera.data.lens=65;camera.data.sensor_width=36
    camera.data.clip_start=.02;camera.data.clip_end=200
    focus=bpy.data.objects.new('Contact plane focus',None);bpy.context.collection.objects.link(focus)
    focus.location=(0,0,-.02)
    camera.data.dof.use_dof=True;camera.data.dof.focus_object=focus;camera.data.dof.aperture_fstop=8.
    tracer_obj=None;tracer_template=None
    if config.get('flow_tracers'):
        count=config['flow_tracers']
        unit=np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]],dtype=float)
        tracer_template=np.tile(unit*.00004*SCALE,(count,1))
        faces=np.array([[0,2,4],[2,1,4],[1,3,4],[3,0,4],[2,0,5],[1,2,5],[3,1,5],[0,3,5]])
        indices=np.concatenate([faces+i*6 for i in range(count)])
        data=bpy.data.meshes.new('Advected surface tracers')
        data.from_pydata(tracer_template.tolist(),[],indices.tolist());data.update()
        tracer_obj=bpy.data.objects.new('Sparse floating tracers',data)
        bpy.context.collection.objects.link(tracer_obj)
        tracer_obj.data.materials.append(material('Reflective microfloat',(.70,.79,.77),.25,.28))
        for polygon in data.polygons:polygon.use_smooth=True
    return scene,water_obj,verts,roots,camera,probe_root,tracer_obj,tracer_template


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--width',type=int,default=1280);p.add_argument('--height',type=int,default=800)
    p.add_argument('--samples',type=int,default=48);p.add_argument('--cpu',action='store_true')
    p.add_argument('--resume',action='store_true',help='Reuse completed PNG frames with matching render settings')
    p.add_argument('--denoiser',choices=['OPENIMAGEDENOISE','OPTIX'],default='OPENIMAGEDENOISE')
    p.add_argument('--stream',action='store_true',help='Wait for sequential simulation states')
    p.add_argument('--frames',type=int,nargs='*');p.add_argument('--stride',type=int,default=1)
    args=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'frames').mkdir(exist_ok=True)
    config=json.loads((args.input/'config.json').read_text());mesh=np.load(args.input/'mesh.npz')
    scene,water,verts,roots,camera,probe,tracers,tracer_template=setup(config,args,mesh)
    files=[args.input/'states'/f'{i:05d}.npz' for i in range(config['frames'])]
    indices=args.frames if args.frames is not None else list(range(0,len(files),args.stride))
    if any(i<0 or i>=len(files) for i in indices):
        raise ValueError('Requested frame is outside the simulation range.')
    if not args.stream:
        missing=[i for i in indices if not files[i].exists()]
        if missing:raise FileNotFoundError(f'Missing simulation frames: {missing[:8]}')
        if args.frames is None and not (args.input/'complete.json').exists():
            raise ValueError('Full rendering requires a completed simulation or --stream.')
    if args.resume and (args.output/'render_config.json').exists():
        previous=json.loads((args.output/'render_config.json').read_text())
        expected=dict(input=str(args.input.resolve()),width=args.width,height=args.height,samples=args.samples,frames=indices)
        if any(previous.get(key)!=value for key,value in expected.items()):
            raise ValueError('Resume requires matching input, frame indices and image settings.')
        if previous.get('denoiser','OPENIMAGEDENOISE')!=args.denoiser:
            raise ValueError('Resume requires the same denoiser.')
    (args.output/'render_config.json').write_text(json.dumps(dict(input=str(args.input.resolve()),
        width=args.width,height=args.height,samples=args.samples,frames=indices,denoiser=args.denoiser,denoising_gpu=not args.cpu,
        renderer='Blender 4.2.9 Cycles / OptiX',scale=SCALE),indent=2))
    started=time.time();records=[]
    n=config['nx']*config['ny']
    for index in indices:
        if args.stream:
            ready=(files[index+1] if index+1<len(files) else args.input/'complete.json')
            deadline=time.monotonic()+600
            while not ready.exists():
                if time.monotonic()>deadline:
                    raise TimeoutError(f'Simulation did not finish frame {index}')
                time.sleep(.5)
        state=np.load(files[index]);t=index/max(config['frames']-1,1)
        verts[:n,2]=state['height'].ravel()*SCALE
        water.data.vertices.foreach_set('co',verts.astype(np.float32).ravel());water.data.update()
        for obj,q,R in zip(roots,state['positions'],state['rotations']):
            mat=Matrix(R.tolist()).to_4x4();mat.translation=Vector(q*SCALE);obj.matrix_world=mat
        if probe is not None:
            probe.location=Vector(state['probe_position']*SCALE)
        if tracers is not None:
            centers=state['tracer_positions'].copy();centers[:,2]+=.000012
            coordinates=tracer_template+np.repeat(centers*SCALE,6,axis=0)
            tracers.data.vertices.foreach_set('co',coordinates.astype(np.float32).ravel())
            tracers.data.update()
        angle=math.radians(-110+48*t)
        distance=5.3 if config['scene']=='capillary_cluster' else 5.7
        elevation=5.1+.3*math.sin(math.pi*t)
        camera.location=(distance*math.cos(angle),distance*math.sin(angle),elevation)
        target=Vector((0,0,-.10))
        if config.get('stirrer') and config['scene']=='capillary_cluster':
            center=state['positions'].mean(axis=0)*SCALE
            offset=center[:2]*.9
            camera.location=(6.0*math.cos(angle)+offset[0],
                             6.0*math.sin(angle)+offset[1],elevation+.75)
            target=Vector((offset[0],offset[1],.45))
        camera.rotation_euler=(target-camera.location).to_track_quat('-Z','Y').to_euler()
        scene.frame_set(index+1)
        path=(args.output/'frames'/f'{index:05d}.png').resolve()
        reuse=False
        if args.resume and path.exists():
            png=path.read_bytes()
            reuse=(png[:8]==b'\x89PNG\r\n\x1a\n' and png[-12:]==b'\x00\x00\x00\x00IEND\xaeB`\x82'
                   and int.from_bytes(png[16:20],'big')==args.width
                   and int.from_bytes(png[20:24],'big')==args.height)
        if not reuse:
            scene.render.filepath=str(path)
            bpy.ops.render.render(write_still=True)
        records.append(dict(frame=index,physical_time=float(state['time']),camera=list(camera.location)))
        (args.output/'frames.json').write_text(json.dumps(records,indent=2))
        print(json.dumps(dict(frame=index,elapsed=round(time.time()-started,2),reused=reuse)),flush=True)
    (args.output/'frames.json').write_text(json.dumps(records,indent=2))
    print('RENDER COMPLETE',args.output,flush=True)


if __name__=='__main__':main()
