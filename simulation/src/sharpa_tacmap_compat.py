"""Add the Isaac Lab 2.3.2 ray-casting APIs required by official Sharpa TacMap.

Call :func:`enable_compat` after starting SimulationApp.  Existing Lab 2.2.1
classes and installation files are left intact.  The newer RayCaster and Warp
kernels live in a private namespace; only missing public APIs are registered.
Vendored Isaac Lab files are unmodified BSD-3-Clause upstream sources.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import json
import sys
import types
from pathlib import Path

_ROOT = Path(__file__).resolve().parent / "sharpa_tacmap_vendor/isaaclab_232"
_SOURCE = _ROOT / "source/isaaclab/isaaclab"
_NAMESPACE = "_sharpa_isaaclab232"
_STATUS = None


def _package(name, path=None):
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__path__ = [] if path is None else [str(path)]
        module.__package__ = name
        sys.modules[name] = module
    return sys.modules[name]


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def enable_compat():
    """Return provenance after making unchanged SharpaTacmap importable.

    Idempotent for a process. Dynamic targets should be rigid bodies or
    identity-transform mesh children of rigid bodies, as in this task. The
    XformPrimView adapter exists for static fallback only; it does not make
    non-physics USD transforms track headless simulation.
    """
    global _STATUS
    if _STATUS is not None:
        return dict(_STATUS)

    import isaaclab
    import isaaclab.sim as sim_utils
    import isaaclab.utils.warp as warp_utils
    ray_package = importlib.import_module("isaaclab.sensors.ray_caster")

    added = []
    if importlib.util.find_spec("isaaclab.sensors.ray_caster.multi_mesh_ray_caster") is None:
        _package(_NAMESPACE)
        _package(_NAMESPACE + ".sensors")
        _package(_NAMESPACE + ".sensors.ray_caster", _SOURCE / "sensors/ray_caster")
        _package(_NAMESPACE + ".warp", _SOURCE / "utils/warp")

        if "isaaclab.sim.views" not in sys.modules and importlib.util.find_spec("isaaclab.sim.views") is None:
            from isaacsim.core.prims import XFormPrim
            from isaacsim.core.utils.stage import get_current_stage

            class XformPrimView(XFormPrim):
                """2.3 constructor vocabulary over Sim 5.0's static XForm view."""
                def __init__(self, prim_path, *, device=None, stage=None):
                    if stage is not None and stage != get_current_stage():
                        raise ValueError("Xform fallback requires the current USD stage")
                    super().__init__(prim_paths_expr=prim_path)

            views = types.ModuleType("isaaclab.sim.views")
            views.XformPrimView = XformPrimView
            sys.modules[views.__name__] = views
            sim_utils.views = views
            added.append("isaaclab.sim.views.XformPrimView (Sim 5.0 adapter)")

        transforms = _load(_NAMESPACE + ".transforms", _SOURCE / "sim/utils/transforms.py")
        for name in ("resolve_prim_pose", "resolve_prim_scale"):
            if not hasattr(sim_utils, name):
                setattr(sim_utils, name, getattr(transforms, name))
                added.append("isaaclab.sim." + name)

        if importlib.util.find_spec("isaaclab.utils.mesh") is None:
            mesh = _load("isaaclab.utils.mesh", _SOURCE / "utils/mesh.py")
            utils_package = importlib.import_module("isaaclab.utils")
            utils_package.mesh = mesh
            added.append("isaaclab.utils.mesh")

        _load(_NAMESPACE + ".warp.kernels", _SOURCE / "utils/warp/kernels.py")
        ops = _load(_NAMESPACE + ".warp.ops", _SOURCE / "utils/warp/ops.py")
        if not hasattr(warp_utils, "raycast_dynamic_meshes"):
            warp_utils.raycast_dynamic_meshes = ops.raycast_dynamic_meshes
            added.append("isaaclab.utils.warp.raycast_dynamic_meshes")

        # New ray-caster behavior is private; existing Lab RayCaster is untouched.
        aliases = {
            _NAMESPACE + ".sensors.sensor_base": "isaaclab.sensors.sensor_base",
            _NAMESPACE + ".sensors.ray_caster.ray_caster_data": "isaaclab.sensors.ray_caster.ray_caster_data",
            _NAMESPACE + ".sensors.ray_caster.ray_caster_cfg": "isaaclab.sensors.ray_caster.ray_caster_cfg",
        }
        for alias, original in aliases.items():
            sys.modules[alias] = importlib.import_module(original)
        ray_dir = _SOURCE / "sensors/ray_caster"
        for name in ("ray_cast_utils", "ray_caster", "multi_mesh_ray_caster_data",
                     "multi_mesh_ray_caster", "multi_mesh_ray_caster_cfg"):
            module = _load(_NAMESPACE + ".sensors.ray_caster." + name, ray_dir / (name + ".py"))
            if name != "ray_caster":
                public_name = "isaaclab.sensors.ray_caster." + name
                sys.modules[public_name] = module
                setattr(ray_package, name, module)
                added.append(public_name)

    manifest = json.loads((_ROOT / "provenance.json").read_text())
    checks = []
    for row in manifest["files"]:
        digest = hashlib.sha256((_ROOT / row["path"]).read_bytes()).hexdigest()
        if digest != row["sha256"]:
            raise RuntimeError("Vendored upstream file hash changed: " + row["path"])
        checks.append({"path": str(_ROOT / row["path"]), "sha256": digest})
    engine_objects = {
        "MultiMeshRayCaster": importlib.import_module(
            "isaaclab.sensors.ray_caster.multi_mesh_ray_caster").MultiMeshRayCaster,
        "obtain_world_pose_from_view": importlib.import_module(
            "isaaclab.sensors.ray_caster.ray_cast_utils").obtain_world_pose_from_view,
        "raycast_dynamic_meshes": warp_utils.raycast_dynamic_meshes,
    }
    engine_modules = {}
    for name, obj in engine_objects.items():
        path = Path(inspect.getfile(obj)).resolve()
        engine_modules[name] = {"module": obj.__module__, "path": str(path),
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if _NAMESPACE + ".warp.kernels" in sys.modules:
        path = Path(sys.modules[_NAMESPACE + ".warp.kernels"].__file__).resolve()
        engine_modules["dynamic_warp_kernels"] = {
            "module": _NAMESPACE + ".warp.kernels", "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    _STATUS = {
        "mode": "isolated_missing_api_backport" if added else "native",
        "active_isaaclab_path": str(Path(isaaclab.__file__).resolve()),
        "active_isaaclab_version": getattr(isaaclab, "__version__", "unknown"),
        "upstream_isaaclab_commit": manifest["commit"],
        "added_apis": added,
        "unmodified_upstream_files": checks,
        "loaded_engine_modules": engine_modules,
        "compatibility_adapter": {"path": str(Path(__file__).resolve()),
                                  "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "system_installation_modified": False,
        "sharpa_source_modified": False,
    }
    return dict(_STATUS)
