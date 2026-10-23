"""Analytically chosen geometry cases for a per-body CPD reference; no simulation."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'analysis/validate_sharpa_tacmap.py').is_file());sys.path.insert(0,str(ROOT/'analysis'))
from validate_sharpa_tacmap import ray_box_depth
def box(x,h):return (np.array([x,0.,0.]),np.array([h,.001,.001]),np.eye(3))
def cpd(p,n,boxes):
 def cast(st,di):return min(float(ray_box_depth(st,di,*b,.5)) for b in boxes)
 first=cast(np.array(p),np.array(n))
 if not np.isfinite(first) or first>.015:return 0.
 second=cast(np.array(p)+(max(first,0)-.0001)*np.array(n),-np.array(n))
 return first if np.isfinite(second) else 0.
cases=[
 ('single_body_interior',[0,0,0],[1,0,0],[[box(0,.001)]],.001,.001,.001,'Single-body reduction preserves the original two-pass result.'),
 ('single_body_exterior',[.003,0,0],[-1,0,0],[[box(0,.001)]],0,0,0,'An outside start is rejected by the closed convex target.'),
 ('two_body_air_gap',[0,0,0],[1,0,0],[[box(-.005,.0005)],[box(.005,.0005)]],.0045,0,0,'Global forward/reverse wrongly pairs opposite bodies; per-body pairing rejects the gap.'),
 ('overlapping_bodies_interior',[0,0,0],[1,0,0],[[box(0,.001)],[box(.0005,.001)]],.001,.001,.0015,'Minimum accepted CPD is nearest single-body exit; maximum is a sensitivity alternative, not the declared main merge.'),
 ('thin_40um_body_interior',[0,0,0],[1,0,0],[[box(0,.00002)]],0,0,0,'The official -100um reverse offset crosses the back side; this limitation is deliberately preserved, without an inside-mask override.'),
 ('one_body_disconnected_components',[0,0,0],[1,0,0],[[box(-.005,.0005),box(.005,.0005)]],.0045,.0045,.0045,'Per-body pairing does not remove ambiguity between disconnected components belonging to one body.'),
]
rows=[]
for name,p,n,groups,global_expected,min_expected,max_expected,note in cases:
 whole=cpd(p,n,[b for group in groups for b in group]);per=[cpd(p,n,g) for g in groups];positive=[x for x in per if x>0];mn=min(positive) if positive else 0.;mx=max(positive) if positive else 0.;passed=np.allclose([whole,mn,mx],[global_expected,min_expected,max_expected],rtol=0,atol=1e-12);assert passed
 rows.append({'case':name,'global_two_pass_CPD_um':whole*1e6,'per_body_CPD_um':[v*1e6 for v in per],'minimum_positive_merge_um':mn*1e6,'maximum_merge_um':mx*1e6,'passed':bool(passed),'interpretation':note})
result={'scope':'Ground-truth analytic box examples; these checks specify semantics rather than validating the full-trajectory implementation.','official_reverse_offset_um':-100,'inside_mask_override':False,'all_passed':all(r['passed'] for r in rows),'rows':rows}
Path(__file__).with_suffix('.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,indent=2))
