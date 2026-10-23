"""Call before importing AppLauncher, omni or pxr; no hardware SDK required."""
import os
import sys
from pathlib import Path


def register_lens_schemas():
    """Register Isaac Sim 5.0 pip-package lens USD schemas before USD starts.

    Also pass ``--enable omni.usd.schema.omni_lens_distortion`` in Kit args.
    This helper supports the pip installation used for the verified project.
    """
    root = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "isaacsim" / "extscache"
    matches = sorted(root.glob("omni.usd.schema.omni_lens_distortion-*/usd_plugins"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one Isaac Sim lens schema package under {root}; found {matches}")
    value = str(matches[0])
    paths = [p for p in os.environ.get("PXR_PLUGINPATH_NAME", "").split(os.pathsep) if p]
    if value not in paths:
        os.environ["PXR_PLUGINPATH_NAME"] = os.pathsep.join([value, *paths])
    return value
