# W01 CAD-matched input surfaces for official Sharpa TacMap

These are derived input maps, **not the unchanged published Sharpa maps or a
hardware taxel calibration**. The official SharpaTacmap ray-casting, two-pass
CPD, quantization and Gaussian filtering implementation remains unchanged.

The original right-hand TH and 4F-origin 240×240 point indexing is retained.
Each valid point is projected to its exact nearest triangle on the current
official W01 `elastomer_HB1_TH.STL` or `elastomer_HB1_4F.STL`. Normals are the
normalized barycentric interpolation of the source mesh's outward vertex
normals. This creates an approximate surface correspondence, not a bijective UV
mapping. No trajectory, contact force, fitted depth or outward offset is used.

Fixed geometric acceptance rules retain only displacements ≤2.5 mm and normal
changes ≤60°. Unsupported points and normals are zero. All originally invalid
cells remain invalid. TH retains 35,419 of 45,211 original valid points; 4F
retains 39,037 of 41,971. The rejected-distance and rejected-normal masks are
saved separately in each correspondence NPZ.

Point NPY units are millimetres; configure `correction_scale=0.001`,
`pts_offsets=0`, and `resolution_step=1`. `manifest.json` records source maps,
current CAD, generator and output hashes. Each correspondence NPZ records all
original valid points, source triangles, barycentric coordinates, projected
points, normal changes and rejection reasons.

The files are geometric sensor inputs. They do not add compliance or establish
that source-mesh penetration is calibrated physical rubber deformation. With
the existing rigid-body physics, TacMap values represent quantized geometric
penetration, not force or pressure. Gaussian filtering occurs after nonlinear
quantization, so displayed codes are not linearly convertible to millimetres.

Regenerate using the project environment:

```bash
.venv-human/bin/python analysis/sharpa_tacmap/build_w01_matched_maps_v2.py --out NEW_DIRECTORY
```

The accompanying original license and notice files retain upstream Sharpa
attribution. See `analysis/sharpa_tacmap/rejected_candidate_history.json` for
earlier draft limitations; neither rejected draft entered a simulation run.
