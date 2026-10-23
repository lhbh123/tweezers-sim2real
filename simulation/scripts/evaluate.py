#!/usr/bin/env python3
"""Run fresh physics trials in independent Isaac processes, at most 3 at once."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--out',default='runs/suite_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))
p.add_argument('--gpus',default='0,1,2')
p.add_argument('--stress',action='store_true',help='Include lower friction and changed timestep; known difficult cases')
args=p.parse_args()
gpus=[int(x) for x in args.gpus.split(',')]
if not 1<=len(gpus)<=3: p.error('Use one to three GPUs to avoid expensive simultaneous Kit startup')
base=ROOT/args.out;base.mkdir(parents=True,exist_ok=True)
cases=[('nominal',[]),('x_plus_2mm',['--offset-x','.002']),('x_minus_2mm',['--offset-x','-.002']),
       ('y_plus_2mm',['--offset-y','.002']),('y_minus_2mm',['--offset-y','-.002']),
       ('block_3g',['--block-mass','.003'])]
if args.stress:
    cases += [('finger_friction_1',['--friction','1.0']),('dt_480',['--dt',str(1/480)])]

def worker(gpu,assigned):
    reports=[]
    for name,options in assigned:
        run=base/name
        command=['/cpfs/user/lvzhaoyang/apps/bin/isaac-python',str(ROOT/'src/run_tweezers.py'),
                 '--out',str(run),'--gpu',str(gpu),*options]
        with (base/(name+'.log')).open('w') as log:
            proc=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        row={'case':name,'command':command,'process_returncode':proc.returncode}
        if (run/'result.json').exists():
            row['result']=json.loads((run/'result.json').read_text())
            validation=subprocess.run([sys.executable,str(ROOT/'scripts/validate_run.py'),str(run)],cwd=ROOT,capture_output=True,text=True)
            row['validation_returncode']=validation.returncode
            row['validated_success']=validation.returncode==0
        else:
            row.update(validated_success=False,error='Simulation produced no result.json')
        (base/(name+'_summary.json')).write_text(json.dumps(row,indent=2))
        print(name,row['validated_success'],flush=True);reports.append(row)
    return reports

with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
    futures=[pool.submit(worker,g,cases[i::len(gpus)]) for i,g in enumerate(gpus)]
    rows=[row for f in futures for row in f.result()]
report={'cases':rows,'success_count':sum(x['validated_success'] for x in rows),'trial_count':len(rows),
        'interpretation':'Finite deterministic parameter checks, not a statistical generalization success rate.'}
(base/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
