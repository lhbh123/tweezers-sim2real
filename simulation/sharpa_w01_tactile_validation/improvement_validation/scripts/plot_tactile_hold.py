"""Plot raw TacMap grayscale, raw CPD depth and separate measured contact force."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run');p.add_argument('--out',required=True)
    args=p.parse_args();run=Path(args.run);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    z=np.load(run/'tacmap.npz');s=np.load(run/'timeseries.npz');report=json.loads((run/'report.json').read_text())
    t=z['frame_time_s'];pt=s['time'];fingers=list(z['finger_names'])
    fig,axs=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    metrics={}
    for col,name in enumerate(('thumb','index')):
        i=fingers.index(name);gray=z['deformation_uint8'][:,i];depth=z['diagnostic_raw_cpd_depth_m'][:,i]
        raw=gray[-1];d=depth[-1]*1e3
        a=axs[0,col].imshow(raw,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
        axs[0,col].set_title(f'{name}: original gray 0..255\nlast peak {raw.max()}');fig.colorbar(a,ax=axs[0,col],fraction=.045)
        b=axs[1,col].imshow(d,cmap='magma',vmin=0,vmax=1,interpolation='nearest')
        axs[1,col].set_title(f'{name}: raw CPD depth (mm)\nlast peak {d.max():.3f} mm');fig.colorbar(b,ax=axs[1,col],fraction=.045)
        for row in range(2):axs[row,col].set_xticks([]);axs[row,col].set_yticks([])
        axs[0,2].plot(t,gray.reshape(len(t),-1).max(-1),label=name)
        jaw=['left','right'].index(report['face_pinch_assignment'][name]);role=['thumb','index'].index(name)
        force=s['broad_face_inward_force_N'][:,0,jaw,role]
        axs[1,2].plot(pt,force,label=name)
        metrics[name]={'last_gray_peak':int(raw.max()),'all_gray_peak':int(gray.max()),
                       'last_cpd_max_depth_mm':float(d.max()),'last_positive_sample_count':int((depth[-1]>0).sum()),
                       'last_assigned_broad_face_inward_force_N':float(force[-1])}
    axs[0,2].set_title('Peak raw gray (no normalization)');axs[0,2].set_ylim(0,255);axs[0,2].set_ylabel('gray code')
    axs[1,2].set_title('Assigned broad-face normal force\n(measured separately from TacMap)',fontsize=11);axs[1,2].set_ylabel('N on the tool')
    for row in range(2):axs[row,2].set_xlabel('Simulation time (s)');axs[row,2].legend();axs[row,2].grid(alpha=.2)
    fig.suptitle(f'{run.name} | final maps at {t[-1]:.3f} s\nGrayscale encodes geometric CPD, not force. Sample counts are not calibrated contact area.',fontsize=14)
    filename=out/(run.name+'_tactile_validation.png');fig.savefig(filename,dpi=160);plt.close(fig)
    result={'run':str(run.resolve()),'figure':str(filename.resolve()),'final_map_time_s':float(t[-1]),
            'grayscale_scale':[0,255],'depth_scale_mm':[0,1],'metrics':metrics,
            'force_definition':'Measured qualified broad-face inward normal component on assigned tool jaw; not total DP merged-link load.',
            'input_sha256':{name:hashlib.sha256((run/name).read_bytes()).hexdigest() for name in ('tacmap.npz','timeseries.npz','report.json')}}
    (out/(run.name+'_tactile_validation.json')).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
