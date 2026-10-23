"""Plot archived contacts against the native convex representation, without simulation."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=Path(__file__).resolve().parent
name='compliant_v1_full_validation_cooked_geometry'
report=json.loads((out/(name+'.json')).read_text())
meta=json.loads((Path(report['geometry_directory'])/'metadata.json').read_text())
rows={r['collider_path']:r for r in report['rows']}
data=np.load(out/(name+'.npz'))
cold=np.load(out/'cold_contact_source_geometry.npz')
fig,ax=plt.subplots(2,3,figsize=(14,8.2))
bins=np.geomspace(.001,1e4,75)
for col,finger in enumerate(['thumb','index','middle']):
    group=[r for r in meta['colliders'] if '/right_'+finger+'_DP/' in r['collider_path']]
    for row in group:
        key=row['key']; kind='skin' if row['surface_class']=='skin' else 'shell'
        for suffix,label,style in [('source','source CAD','--'),('any_hull','native cooked','-')]:
            a=data[key+'_contact_to_'+suffix+'_um']
            ax[0,col].hist(np.maximum(a,.001),bins=bins,histtype='step',linestyle=style,label=kind+' '+label)
    ax[0,col].set_xscale('log');ax[0,col].set_title(finger+' : archived cold normal contacts')
    ax[0,col].set_xlabel('Closest surface distance (µm)');ax[0,col].set_ylabel('Point count')
    ax[0,col].legend(fontsize=8);ax[0,col].grid(alpha=.2)
    selected=next(r for r in group if r['surface_class']==('non_skin' if finger=='middle' else 'skin'))
    key=selected['key']; times=cold[finger+'_time_s']
    for suffix,label in [('source','source CAD'),('any_hull','native cooked')]:
        ax[1,col].scatter(times,np.maximum(data[key+'_contact_to_'+suffix+'_um'],.001),s=7,label=label,alpha=.6)
    ax[1,col].set_yscale('log');ax[1,col].set_ylim(.001,1000)
    ax[1,col].set_title('Nearest '+('shell' if finger=='middle' else 'skin')+' : same contact points')
    ax[1,col].set_xlabel('Cold-run time (s)');ax[1,col].set_ylabel('Closest surface distance (µm)')
    ax[1,col].grid(alpha=.2);ax[1,col].legend(fontsize=8)
fig.suptitle('Actual contact locations agree with the exported convex geometry',fontsize=16)
fig.text(.5,.03,'Cold_v3 contacts in body coordinates vs final-run native cooking representation; geometry-only evidence, not solver collider IDs.',ha='center',fontsize=10)
fig.text(.5,.01,'Cooked distances include internal hull faces. Thumb/index contacts are near skin; middle contacts are near shell.',ha='center',fontsize=10)
fig.tight_layout(rect=[0,.06,1,.95]);fig.savefig(out/(name+'.png'),dpi=150)
