from pathlib import Path
import argparse,base64,concurrent.futures,hashlib,json,urllib.request

p=argparse.ArgumentParser();p.add_argument('--api',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parents[2]/'src/sharpa_tacmap_vendor/isaaclab_232'
commit='37ddf626871758333d6ed89cf64ad702aef127d0'
files=['source/isaaclab/isaaclab/sensors/ray_caster/'+n+'.py'for n in ['multi_mesh_ray_caster','multi_mesh_ray_caster_cfg','multi_mesh_ray_caster_data','ray_cast_utils','ray_caster']]+['source/isaaclab/isaaclab/utils/warp/'+n+'.py'for n in ['ops','kernels']]+['source/isaaclab/isaaclab/utils/mesh.py','source/isaaclab/isaaclab/sim/utils/transforms.py','LICENSE']
def one(name):
 try:
  url=(f'https://api.github.com/repos/isaac-sim/IsaacLab/contents/{name}?ref={commit}'if a.api else f'https://raw.githubusercontent.com/isaac-sim/IsaacLab/{commit}/{name}')
  req=urllib.request.Request(url,headers={'User-Agent':'local-compat-audit'})
  content=urllib.request.urlopen(req,timeout=25).read()
  if a.api:content=base64.b64decode(json.loads(content)['content'])
  dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(content)
  return {'path':name,'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content),'url':url}
 except Exception as e:return {'path':name,'error':str(e)}
rows=list(concurrent.futures.ThreadPoolExecutor(7).map(one,files));root.mkdir(parents=True,exist_ok=True);(root/'provenance.json').write_text(json.dumps({'upstream':'https://github.com/isaac-sim/IsaacLab','commit':commit,'files':rows},indent=2)+'\n');print(json.dumps(rows,indent=2))
