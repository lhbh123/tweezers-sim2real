"""Read live PhysX material values; import only after the scene is initialized.

This module does not step simulation, author USD, or set any physics properties.
Use capture_scene_materials(scene, out_path) immediately after reset. Public
tensor get_material_properties returns [static friction, dynamic friction,
restitution]; USD schema fallbacks are recorded separately and never labelled
as effective runtime values. Shape-to-collider ordering is not assumed.
"""
import json
from pathlib import Path


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu()
    if hasattr(value, 'numpy'):
        value = value.numpy()
    return value.tolist()


def capture_view(view, label):
    item = {'label': label, 'columns': ['static_friction', 'dynamic_friction', 'restitution']}
    for attr in ('prim_paths', 'link_paths', 'count', 'max_shapes'):
        try:
            value = getattr(view, attr)
            item[attr] = value if isinstance(value, (list, str, int, float)) else list(value)
        except Exception as exc:
            item[attr + '_unavailable'] = str(exc)
    for method in ('get_material_properties', 'get_contact_offsets', 'get_rest_offsets'):
        try:
            item[method] = _array(getattr(view, method)())
        except Exception as exc:
            item[method + '_error'] = str(exc)
    return item


def capture_scene_materials(scene, out_path):
    from pxr import Usd, UsdPhysics, UsdShade, PhysxSchema
    import omni.physics.tensors as tensors
    root = scene.env_paths[0] + '/Robot'
    stage = scene.stage
    # Numpy frontend is rejected by OmniPhysics when the scene uses the GPU
    # pipeline. Torch supports both pipelines; copy results to CPU for JSON.
    sim_view = tensors.create_simulation_view('torch')
    records = []
    # One view per rigid body makes shape ownership explicit. Each body's
    # remaining shape order is deliberately left anonymous.
    paths = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and ('_DP' in prim.GetName()):
            paths.append(str(prim.GetPath()))
    paths += [scene.env_paths[0] + '/Tweezers/' + x for x in ('Handle', 'Left', 'Right')]
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if not prim:
            continue
        try:
            view = sim_view.create_rigid_body_view(path)
            item = capture_view(view, path)
        except Exception as exc:
            item = {'label': path, 'runtime_error': str(exc)}
        colliders = []
        for child in Usd.PrimRange(prim):
            if not child.HasAPI(UsdPhysics.CollisionAPI):
                continue
            entry = {'collider_path': str(child.GetPath())}
            try:
                material, relation = UsdShade.MaterialBindingAPI(child).ComputeBoundMaterial('physics')
                entry['material_path'] = str(material.GetPath()) if material else None
                if material:
                    mp = material.GetPrim()
                    entry['has_PhysicsMaterialAPI'] = mp.HasAPI(UsdPhysics.MaterialAPI)
                    entry['has_PhysxMaterialAPI'] = mp.HasAPI(PhysxSchema.PhysxMaterialAPI)
                    entry['usd_attributes'] = {}
                    for attr_name in ('physics:staticFriction', 'physics:dynamicFriction',
                                      'physics:restitution', 'physxMaterial:compliantContactStiffness',
                                      'physxMaterial:compliantContactDamping',
                                      'physxMaterial:frictionCombineMode'):
                        attr = mp.GetAttribute(attr_name)
                        entry['usd_attributes'][attr_name] = {
                            'value': attr.Get() if attr else None,
                            'authored': bool(attr and attr.HasAuthoredValueOpinion())}
            except Exception as exc:
                entry['usd_error'] = str(exc)
            colliders.append(entry)
        item['usd_colliders_unordered'] = colliders
        records.append(item)
    result = {
        'source': 'omni.physics.tensors live backend get_material_properties after scene reset',
        'read_only': True,
        'notes': [
            'Friction is a live per-shape property; this is not a measured effective contact-pair coefficient.',
            'Collision shape ordering is not mapped to USD collider ordering.',
            'Compliance and combine-mode USD values are authored/schema evidence only; this tensor API does not read them.',
            'To assess the official compatibility spawner, capture the actual official replay, not just this task scene.'
        ],
        'bodies': records}
    Path(out_path).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result
