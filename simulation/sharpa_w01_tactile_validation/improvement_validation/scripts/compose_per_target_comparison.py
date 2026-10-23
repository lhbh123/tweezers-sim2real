"""CPU-only media for saved official vs per-target TacMap observations.

This never changes the physical trajectory or replaces the original archive.
The paired-body reference is explicitly labelled as a diagnostic extension.
"""
from pathlib import Path
import argparse
import hashlib
import json

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'sharpa_w01_tactile_validation/improvement_validation'
FONT = ROOT/'assets/fonts/NotoSansCJKsc-Regular.otf'
BG = (10, 17, 28)
WHITE, MUTED = (232, 240, 250), (151, 172, 196)
ORANGE, CYAN = (255, 180, 78), (66, 212, 201)
PHASES = ('初始持镊', '接近方块', '下降对准', '挤压夹紧', '提起方块',
          '搬运至目标', '下降放置', '松开镊尖', '撤离镊尖', '稳定留置')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


class Media:
    def __init__(self, old, paired, times, rows, steps, phase):
        self.old, self.paired, self.times = old, paired, times
        self.rows, self.steps, self.phase, self.fonts = rows, steps, phase, {}
        self.peaks = np.stack([a.max((1, 2)) for a in [old, paired]])
        self.positive = np.stack([np.count_nonzero(a, axis=(1, 2)) for a in [old, paired]])

    def text(self, draw, xy, value, size=22, colour=WHITE, width=None):
        for chosen in range(size, 13, -1):
            if chosen not in self.fonts:
                self.fonts[chosen] = ImageFont.truetype(str(FONT), chosen)
            if width is None or draw.textlength(str(value), font=self.fonts[chosen]) <= width:
                draw.text(xy, str(value), font=self.fonts[chosen], fill=colour)
                return
        raise ValueError('Text cannot fit assigned layout region: '+str(value))

    def frame(self, camera, i):
        frame = Image.new('RGB', (1280, 912), BG)
        draw = ImageDraw.Draw(frame)
        self.text(draw, (25, 14), '同一物理轨迹：官方算法原输出与逐目标旁路', 32, width=1230)
        self.text(draw, (25, 58), '离线观测对照 · 旁路不是原版官方输出，也不用作任务成功证据', 21, CYAN, 1230)
        self.text(draw, (25, 99), '原生任务画面（与两路触觉同一时刻）', 20, MUTED)
        pic = Image.fromarray(np.asarray(camera, np.uint8)[:, :, :3])
        pic.thumbnail((590, 442), Image.Resampling.LANCZOS)
        draw.rectangle((25, 132, 615, 574), fill=(0, 0, 0))
        frame.paste(pic, (25+(590-pic.width)//2, 132+(442-pic.height)//2))
        self.text(draw, (650, 99), '食指：固定 0–255 灰度，均不做亮度拉伸', 20, MUTED, 600)
        for k, (x, title, values, colour) in enumerate([
            (650, '本任务官方算法原输出', self.old[i], ORANGE),
            (965, '离线逐目标 / 最小正CPD', self.paired[i], CYAN),
        ]):
            self.text(draw, (x, 143), title, 22, colour, 280)
            # Preserve the upstream uint8 values on a fixed range, with nearest resize.
            p = Image.fromarray(values, mode='L').convert('RGB').resize((280, 280), Image.Resampling.NEAREST)
            frame.paste(p, (x, 181))
            self.text(draw, (x, 471), f'灰度峰值 {int(values.max())} / 255', 21, colour)
            self.text(draw, (x, 502), f'非零灰度像素 {np.count_nonzero(values)}', 18, MUTED)
            raw_key = 'original_raw_peak_mm' if k == 0 else 'min_raw_peak_mm'
            self.text(draw, (x, 531), f'几何 CPD 峰值 {self.rows[i][raw_key]:.3f} mm', 18, MUTED, 280)
        gradient = np.tile(np.arange(256, dtype=np.uint8), (8, 1))
        frame.paste(Image.fromarray(gradient).convert('RGB').resize((595, 8), Image.Resampling.NEAREST), (650, 565))
        self.text(draw, (650, 575), '0', 14, MUTED)
        self.text(draw, (1215, 575), '255', 14, MUTED)
        phase = int(self.phase[int(self.steps[i])].reshape(-1)[0])
        self.text(draw, (25, 584), f'{self.times[i]:.3f} s  |  {PHASES[phase]}', 23, WHITE, 590)
        self.text(draw, (25, 630), '食指全轨迹灰度峰值', 22)
        self.text(draw, (690, 632), '— 本任务原始', 20, ORANGE)
        self.text(draw, (925, 632), '— 逐目标旁路', 20, CYAN)
        x0, y0, x1, y1 = 75, 680, 1220, 815
        t0, t1 = float(self.times[0]), float(self.times[-1])
        xs = x0+(self.times-t0)/(t1-t0)*(x1-x0)
        for val in [0, 128, 255]:
            y = y1-val/255*(y1-y0)
            draw.line((x0, y, x1, y), fill=(42, 54, 68), width=1)
            self.text(draw, (28, y-10), val, 16, MUTED)
        for values, colour in zip(self.peaks, [ORANGE, CYAN]):
            ys = y1-values.astype(float)/255*(y1-y0)
            draw.line(list(zip(xs.tolist(), ys.tolist())), fill=colour, width=2)
        current_x = float(xs[i]);draw.line((current_x, y0, current_x, y1), fill=WHITE, width=2)
        for t in [0, 5, 10, 15, 20, 25]:
            x = x0+(t-t0)/(t1-t0)*(x1-x0)
            if x0-2 <= x <= x1+2:self.text(draw, (x-10, y1+4), f'{t}s', 15, MUTED)
        self.text(draw, (25, 852), '官方原图含跨镊臂射线误接受；旁路将两次射线限制在同一目标，再取最小正 CPD。', 19, ORANGE, 1230)
        self.text(draw, (25, 880), '两路都不代表实测形变、力或压力；旁路仍保留薄物体及同一刚体不连通几何的限制。', 18, MUTED, 1230)
        return frame


def summary_figure(media, camera, representative, report, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(FONT)).get_name(),
                         'axes.unicode_minus':False, 'font.size':11})
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.4), facecolor='white', gridspec_kw={'height_ratios':[1.1, 1]})
    row = media.rows[representative]
    axes[0,0].imshow(camera);axes[0,0].set_title(f'同一原生场景 · {media.times[representative]:.3f} s')
    for ax, values, label, rawkey in [
        (axes[0,1],media.old[representative],'本任务官方算法原输出','original_raw_peak_mm'),
        (axes[0,2],media.paired[representative],'逐目标旁路（非原版官方）','min_raw_peak_mm'),
    ]:
        ax.imshow(values,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
        ax.set_title(f'食指 {label}\n灰度峰值 {values.max()}；几何CPD峰值 {row[rawkey]:.3f} mm')
    for ax in axes[0]:ax.set_axis_off()
    for ax, values, title, ylabel in [
        (axes[1,0],media.peaks,'完整轨迹：食指灰度峰值','灰度代码（固定0–255）'),
        (axes[1,1],media.positive,'完整轨迹：食指非零灰度像素','像素数（不是接触面积）'),
    ]:
        for series,label,colour in zip(values,['本任务原始','逐目标旁路'],['#d08931','#269a91']):ax.plot(media.times,series,label=label,color=colour,lw=1.8)
        ax.axvline(media.times[representative],color='#687585',ls='--',lw=1)
        ax.set(title=title,xlabel='保存轨迹时间 / s',ylabel=ylabel)
        ax.grid(alpha=.2);ax.legend(fontsize=10)
    axes[1,0].set_ylim(0,255)
    axes[1,2].set_axis_off()
    orig_out = row['original_raw_positive_outside_all_boxes_by_more_than_tolerance_count']
    own_out = sum(b['positive_outside_own_boxes_by_more_than_tolerance_count'] for b in row['bodies'])
    msg=(f'代表帧 {row["source_frame"]}：\n'
         f'原图CPD正值点：{row["original_raw_positive_count"]:,}\n'
         f'其中起点在所有目标盒外 >1 μm：{orig_out:,}\n'
         f'旁路CPD正值点：{row["min_positive_count"]:,}\n'
         f'旁路逐目标盒外 >1 μm 正值点：{own_out:,}\n\n'
         '旁路：每个刚体独立执行两次射线，\n'
         '再取最小正CPD，沿用量化与模糊。\n'
         '这是观测诊断，不是原版官方输出；\n'
         '不改变轨迹，也不作任务成功证据。\n\n'
         '毫米值是几何量，不是实测皮肤形变。\n'
         '同刚体不连通几何、薄物体限制仍在。')
    axes[1,2].text(0,1,msg,va='top',fontsize=11,linespacing=1.45)
    fig.suptitle('本任务跨镊臂射线误接受：原始与逐目标离线适配对照',fontsize=21)
    fig.text(.5,.018,'同一完整物理轨迹；灰度统一0–255，未拉伸、未裁剪异常像素、未修改官方归档。',ha='center',fontsize=11,color='#526174')
    fig.tight_layout(rect=[0,.045,1,.95]);fig.savefig(out,dpi=170);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=BASE/'runs/compliant_v1_full_validation_v2')
    p.add_argument('--replay',type=Path,default=BASE/'per_target_replay/full_trajectory')
    p.add_argument('--output',type=Path,default=BASE/'visualizations/W01_官方与逐目标旁路对照.mp4')
    p.add_argument('--max-mb',type=float,default=10.)
    p.add_argument('--representative-frame',type=int,default=399)
    p.add_argument('--overwrite',action='store_true')
    args=p.parse_args()
    if args.output.exists() and not args.overwrite:raise FileExistsError(args.output)
    replay_report=json.loads((args.replay/'report.json').read_text())
    if replay_report['state']!='complete':raise ValueError('Replay report is not complete')
    if replay_report['source_tacmap_npz_sha256']!=sha(args.run/'tacmap.npz'):raise ValueError('Replay is from another TacMap archive')
    with np.load(args.replay/'per_target_tacmap.npz') as z:
        names=z['finger_names'].tolist();k=names.index('index')
        old=z['original_gray_uint8'][:,k].copy();paired=z['min_positive_gray_uint8'][:,k].copy()
        times=z['frame_time_s'];steps=z['sample_control_step'];frames=z['source_frame_index']
    with np.load(args.run/'tacmap.npz') as z:
        if not np.array_equal(old,z['deformation_uint8'][frames,k]):raise ValueError('Official original values changed in replay archive')
        if not np.array_equal(times,z['frame_time_s'][frames]):raise ValueError('Replay times differ from source')
    if old.dtype!=np.uint8 or paired.dtype!=np.uint8 or old.shape!=paired.shape:raise ValueError('Expected matched uint8 maps')
    count=len(times)
    if not np.array_equal(frames,np.arange(count)):raise ValueError('Media expects a complete ordered source trajectory')
    with np.load(args.run/'trajectory.npz') as z:
        phase=z['phase'];render_steps=z['render_control_steps']
        if not np.array_equal(steps,render_steps):raise ValueError('Replay/render control steps differ')
    rows=[r for r in replay_report['rows'] if r['finger']=='index']
    if [r['source_frame'] for r in rows]!=frames.tolist():raise ValueError('Report frame rows are not aligned')
    media=Media(old,paired,times,rows,steps,phase)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    source_video=args.run/'task.mp4';reader=imageio.get_reader(source_video)
    fps=float(reader.get_meta_data()['fps'])
    if reader.count_frames()!=count or abs(fps-30.)>1e-6:raise ValueError('Native video frame count/rate differs')
    bitrate=int(args.max_mb*1e6*8*.87/(count/fps))
    representative=args.representative_frame
    if not 0<=representative<count:raise ValueError('Invalid representative frame')
    keys={0,count-1,representative,int(media.peaks[0].argmax())}
    writer=imageio.get_writer(args.output,fps=fps,codec='libx264',bitrate=str(bitrate),macro_block_size=16,
                               ffmpeg_params=['-movflags','+faststart'])
    try:
        for i in range(count):
            camera=reader.get_data(i);frame=media.frame(camera,i);writer.append_data(np.asarray(frame))
            if i in keys:frame.save(args.output.with_name(args.output.stem+f'_frame_{i:05d}.png'))
            if i==representative:
                frame.save(args.output.with_name(args.output.stem+'_cover.png'))
                summary_figure(media,camera,i,replay_report,args.output.with_suffix('.png'))
    finally:
        writer.close();reader.close()
    with imageio.get_reader(args.output) as r:
        dimensions=list(r.get_meta_data()['size']);decoded=r.count_frames()
    if dimensions!=[1280,912] or decoded!=count:raise ValueError('Encoded dimensions/frame count differ')
    size=args.output.stat().st_size
    if size>args.max_mb*1e6:raise ValueError('Video exceeds delivery budget')
    result={'source_run':str(args.run.resolve()),'source_replay':str(args.replay.resolve()),
        'source_video_sha256':sha(source_video),'source_replay_report_sha256':sha(args.replay/'report.json'),
        'source_tacmap_sha256':sha(args.run/'tacmap.npz'),'source_replay_npz_sha256':sha(args.replay/'per_target_tacmap.npz'),
        'source_original_index_uint8_exact_match':True,'same_source_times_and_steps':True,
        'original_gray_peak':int(old.max()),'paired_gray_peak':int(paired.max()),
        'frames':count,'fps':fps,'dimensions':dimensions,'bytes':size,'sha256':sha(args.output),
        'representative_frame':representative,'representative_time_s':float(times[representative]),
        'semantics':'Same saved physical trajectory and native video. Original official uint8 and offline per-target minimum-positive paired-ray reference use fixed 0–255 grayscale and nearest resize. The reference is not original official output, not measured deformation, and not evidence of task success. No force/controller/physics changes or additional simulation.',
        'source_script_sha256':sha(Path(__file__))}
    args.output.with_suffix('.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
