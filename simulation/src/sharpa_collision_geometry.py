"""Snapshot native PhysX convex collision representations without physics writes.

Call after scene reset. The cooking API may populate the mesh cache, but this
module does not author USD, change approximation/settings, step simulation, or
write body/joint state. Exported hull order is not a stable shape identifier.
"""
from pathlib import Path
import hashlib
import json
import numpy as np


def _triangulate(indices, counts):
    triangles=[];offset=0
    for count in counts:
        polygon=indices[offset:offset+count]
        triangles.extend([[polygon[0],polygon[j],polygon[j+1]] for j in range(1,count-1)])
        offset+=count
    return np.asarray(triangles,dtype=np.int32).reshape(-1,3)


def capture_collision_geometry(scene, output_directory, *, phase='after_reset_before_steps'):
    from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils, PhysicsSchemaTools
    from omni.physx import get_physx_cooking_interface
    stage=scene.stage;output=Path(output_directory);output.mkdir(parents=True,exist_ok=True)
    cache=UsdUtils.StageCache.Get();stage_id=cache.GetId(stage);inserted=False
    if not stage_id.IsValid():stage_id=cache.Insert(stage);inserted=True
    cooking=get_physx_cooking_interface();arrays={};rows=[]
    robot=stage.GetPrimAtPath(scene.env_paths[0]+'/Robot')
    for prim in Usd.PrimRange(robot):
        path=str(prim.GetPath())
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or not prim.IsA(UsdGeom.Mesh):continue
        if not any('right_'+f+'_DP' in path or 'right_'+f+'_elastomer' in path for f in ['thumb','index','middle','ring','pinky']):continue
        key='collider_'+str(len(rows));mesh=UsdGeom.Mesh(prim);points=np.asarray(mesh.GetPointsAttr().Get(),dtype=np.float64);indices=np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),dtype=np.int32);counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get(),dtype=np.int32)
        arrays[key+'_source_vertices']=points;arrays[key+'_source_triangles']=_triangulate(indices,counts)
        body=prim
        while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):body=body.GetParent()
        transform=UsdGeom.XformCache(Usd.TimeCode.Default()).ComputeRelativeTransform(prim,body)[0] if body else None
        attributes={}
        for attr in prim.GetAttributes():
            if attr.GetName().startswith(('physics:','physxCollision:','physxConvex')):
                value=attr.Get();attributes[attr.GetName()]={'value':value if isinstance(value,(str,int,float,bool,type(None))) else str(value),'authored':attr.HasAuthoredValueOpinion()}
        row={'key':key,'collider_path':path,'rigid_body_path':str(body.GetPath()) if body else None,
             'surface_class':'skin' if 'elastomer' in path.lower() else 'non_skin',
             'source_vertex_count':len(points),'source_triangle_count':len(arrays[key+'_source_triangles']),
             'mesh_to_rigid_matrix_row_vector':np.asarray(transform).tolist() if transform is not None else None,
             'attributes':attributes,'hulls':[]}
        result_holder={}
        def completed(result,convexes):result_holder.update(result=str(result),convexes=convexes)
        try:
            cooking.request_convex_collision_representation(stage_id=stage_id.ToLongInt(),collision_prim_id=PhysicsSchemaTools.sdfPathToInt(prim.GetPath()),run_asynchronously=False,on_result=completed)
            row['result']=result_holder.get('result','callback_not_received')
            for i,hull in enumerate(result_holder.get('convexes',[])):
                hkey=key+'_hull_'+str(i);vertices=np.asarray([[v.x,v.y,v.z] for v in hull.vertices],dtype=np.float64)
                polygons=[list(hull.indices[p.index_base:p.index_base+p.num_vertices]) for p in hull.polygons]
                flat=np.asarray([index for polygon in polygons for index in polygon],dtype=np.int32);sizes=[len(p) for p in polygons]
                arrays[hkey+'_vertices']=vertices;arrays[hkey+'_triangles']=_triangulate(flat,sizes)
                row['hulls'].append({'key':hkey,'vertices':len(vertices),'triangles':len(arrays[hkey+'_triangles']),'polygon_vertex_counts':sizes})
        except Exception as exc:row['error']=str(exc)
        rows.append(row)
    archive=output/'meshes.npz';np.savez_compressed(archive,**arrays)
    result={'schema_version':1,'phase':phase,'stage_identifier':stage.GetRootLayer().identifier,
        'stage_cache_inserted_for_query':inserted,'no_physics_state_or_parameter_writes':True,
        'method':'omni.physx.get_physx_cooking_interface().request_convex_collision_representation, synchronous callback',
        'coordinate_frame':'Vertices are collider mesh-local, as consumed by the installed NVIDIA ConvexMeshDataDemo; fixed mesh-to-rigid row-vector transforms are also recorded.',
        'limitations':['The query may trigger cooking or fill the cache; it does not change approximation parameters or advance physics.',
            'This is the representation returned for the current stage, not a direct dump of each resident PxShape object.',
            'Convex hull callback order is not stable and must not be matched to material tensor shape indices.',
            'Triangulation fans each convex polygon; internal hull surfaces overlap and should not be confused with the exterior union boundary.'],
        'installed_reference':'omni.physxdemos/scenes/ConvexMeshDataDemo.py',
        'mesh_archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'colliders':rows}
    (output/'metadata.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result
