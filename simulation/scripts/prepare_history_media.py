"""Make smaller H.264 copies of actual delivered videos for an offline HTML report."""
import hashlib,json,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import imageio_ffmpeg

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'reports/conversation_history_media';DEST.mkdir(exist_ok=True)
SOURCES={
 'baseline':'ur5e_xhand_tweezers_demo',
 'human_ppo':'ur5e_xhand_tweezers_ppo_natural',
 'approach':'ur5e_xhand_tweezers_ppo_approach',
 'palm_down':'ur5e_xhand_tweezers_palm_down_delivery',
 'tripod':'ur5e_xhand_tripod_rl_reference_comparison',
 'broad_face':'ur5e_xhand_straight_face_ppo',
 'backup_test':'ur5e_xhand_middle_backup_catch_v14',
 'backup_ppo':'ur5e_xhand_middle_backup_ppo_v14',
 'random_close':'ur5e_xhand_random_pick_place_closeup_v15',
 'random_four':'ur5e_xhand_random_pick_place_four_corners_v15',
}
ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
def work(item):
 key,name=item;src=ROOT/'outputs'/f'{name}.mp4';dst=DEST/f'{key}.mp4'
 if not dst.exists():
  subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-y','-i',str(src),'-map','0:v:0',
   '-vf','scale=min(1280\\,iw):-2','-c:v','libx264','-preset','medium','-crf','26','-threads','4',
   '-pix_fmt','yuv420p','-an','-movflags','+faststart',str(dst)],check=True)
 reader=imageio_ffmpeg.read_frames(str(dst));meta=next(reader);n=sum(1 for _ in reader)
 src_reader=imageio_ffmpeg.read_frames(str(src));src_meta=next(src_reader);source_n=sum(1 for _ in src_reader)
 assert n==source_n and meta['fps']==src_meta['fps']
 poster=DEST/f'{key}.jpg'
 # Preserve the existing selected cover, including its measurement overlays.
 for suffix in ['_cover.png','.png']:
  cover=ROOT/'outputs'/f'{name}{suffix}'
  if cover.exists():break
 from PIL import Image
 if cover.exists():
  im=Image.open(cover).convert('RGB');im.thumbnail((1280,1280));im.save(poster,quality=90)
 else:
  subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-y','-ss',str(meta['duration']/2),
    '-i',str(dst),'-frames:v','1',str(poster)],check=True)
 result=dict(id=key,source=str(src.relative_to(ROOT)),report_copy=str(dst.relative_to(ROOT)),
  poster=str(poster.relative_to(ROOT)),source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
  report_sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),bytes=dst.stat().st_size,
  duration_s=meta['duration'],fps=meta['fps'],frames=n,source_frames=source_n,
  report_resolution=meta['size'],source_resolution=src_meta['size'],
  operation='H264 compression and resizing only; complete original frame sequence and playback speed preserved')
 print(key,n,dst.stat().st_size,flush=True);return result
with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(work,SOURCES.items()))
(DEST/'manifest.json').write_text(json.dumps(rows,indent=2))
print('TOTAL_BYTES',sum(x['bytes'] for x in rows),flush=True)
