from pathlib import Path
import json,hashlib,subprocess,concurrent.futures,time
ROOT=Path(__file__).resolve().parents[2]
AUDIT=ROOT/'analysis/sharpa_research'
DEST=ROOT/'assets/sharpa_wave/official'
COMMIT='0d19cac602f46456b819e4b6a2c09a74982c9a3e'
PROXY='/cpfs/user/lvzhaoyang/office/feishu/飞书连接方案_阿里云4090集群/scripts/bin/with-general-proxy'
rows=json.loads((AUDIT/'sharpa-urdf-usd-xml_tree.json').read_text())['tree']
import xml.etree.ElementTree as ET
urdf=ET.parse(DEST/'wave_01/right_sharpa_wave/right_sharpa_wave_with_flange.urdf')
required_mesh={m.get('filename').replace('package://right_sharpa_wave/','wave_01/right_sharpa_wave/') for m in urdf.findall('.//mesh')}
def selected(r):
 p=r['path']
 return r['type']=='blob' and (p in ['LICENSE.txt','NOTICE.txt','README.md'] or p in required_mesh or p.startswith('wave_01/right_sharpa_wave/right_sharpa_wave_with_flange/') or p in ['wave_01/right_sharpa_wave/right_sharpa_wave_with_flange.urdf','wave_01/right_sharpa_wave/right_sharpa_wave_with_flange.usda','wave_01/right_sharpa_wave/right_sharpa_wave_with_flange_MITmode.usda','wave_01/right_sharpa_wave/right_sharpa_wave_with_flange.xml'])
def valid(p,r):
 if not p.exists():return False
 b=p.read_bytes();return len(b)==r['size'] and hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==r['sha']
def get(r):
 p=DEST/r['path'];p.parent.mkdir(parents=True,exist_ok=True)
 if valid(p,r):print('VERIFIED',r['path'],flush=True);return {'path':r['path'],'status':'existing_verified','bytes':r['size'],'git_blob_sha':r['sha']}
 url=f'https://raw.githubusercontent.com/sharpa-robotics/sharpa-urdf-usd-xml/{COMMIT}/{r["path"]}'
 tmp=Path(str(p)+'.partial')
 # Prior raw direct connection failed; retry the same host through the server's approved wrapper.
 proc=subprocess.run([PROXY,'curl','--fail','--silent','--show-error','--connect-timeout','15','--max-time','120',url,'-o',str(tmp)],capture_output=True,text=True)
 if proc.returncode or not valid(tmp,r):
  print('FAILED',r['path'],proc.returncode,proc.stderr.strip(),flush=True)
  return {'path':r['path'],'status':'failed','returncode':proc.returncode,'message':proc.stderr.strip()}
 tmp.replace(p);print('OK',r['path'],r['size'],flush=True)
 return {'path':r['path'],'status':'downloaded_verified','bytes':r['size'],'git_blob_sha':r['sha'],'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
 for f in concurrent.futures.as_completed([pool.submit(get,r) for r in rows if selected(r)]):results.append(f.result())
manifest={'repository':'https://github.com/sharpa-robotics/sharpa-urdf-usd-xml','commit':COMMIT,'model':'W01','downloaded_at_epoch':time.time(),'required_files':len(results),'all_verified':all(x['status']!='failed' for x in results),'files':sorted(results,key=lambda x:x['path'])}
(DEST/'DOWNLOAD_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
print('DOWNLOAD_RESULT',manifest['all_verified'],len(results),flush=True)
