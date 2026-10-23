"""Build a standalone, offline, evidence-linked Chinese experiment history."""
import base64,datetime,hashlib,html,io,json,re
from pathlib import Path
from PIL import Image
from fontTools import subset
from fontTools.ttLib import TTFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/tweezers_对话与实验演进记录_20261019.html'
def read(p):return json.loads((ROOT/p).read_text())
def esc(x):return html.escape(str(x),quote=True)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def uri(p,mime):return 'data:'+mime+';base64,'+base64.b64encode(p.read_bytes()).decode()
stages=read('reports/conversation_history_content.json')['stages']
media={x['id']:x for x in read('reports/conversation_history_media/manifest.json')}
history=read('reports/history_lark_messages_20261019.json')
assert history['ok'] and history['meta']['pagination']['complete']
messages=history['data']['messages'];begin=False;users=[];replies=[];stops=[];source_messages=[]
for m in messages:
 if m['message_id']=='om_x100b623c3b07b0a8c040cf1e46021b6':begin=True
 if not begin:continue
 if m['message_id']=='om_x100b6268ab842c84cf6e6cdf686f849':
  users.append(m);source_messages.append(m);break
 if m['sender']['sender_type']=='user':
  if m['content']=='/stop':stops.append(m)
  elif not m['content'].startswith('/'):users.append(m);source_messages.append(m)
 elif m['msg_type']=='post' and '当前会话思考强度' not in m['content']:
  replies.append(m);source_messages.append(m)
assert len(users)==18
annotations=[
 ('s1','建立脚本控制基线，完成道具、IK、接触仿真与两段交付视频；六个确定性案例通过。'),
 ('s1','明确答复：当时是状态机、IK 和位姿反馈，没有训练强化学习策略。'),
 ('s2','下载人类视频、提取二维姿态先验、优化 XHand 接触，并真正训练 PPO；先后得到 48/64 与接近版 58/64 的分条件结果。'),
 ('s2','17:37 的错误卡记录了远程上下文压缩连接超时；不是已完成训练后停止。'),
 ('s2','18:03 再次记录连接超时。20:24 恢复时核对了已下载视频和 7 帧姿态结果，承认 PPO 尚未开始；随后完成稳持、训练与验证。'),
 ('s2','确认身份为“Codex阿里云4090”，同时继续此前未完成的人类参考与强化学习任务。'),
 ('s2','先提供 Penn 与 Texas A&M 的来源说明；前者用于姿态先验，后者仅作补充视觉参考。'),
 ('s2','按要求改为直接发送两张真实参考截图：Penn 3:42、Texas A&M 0:30。'),
 ('s3','交付三张姿态解释/对比图，审计二维先验、接触优化和实际执行的关系；这轮没有更换策略。'),
 ('s4','一起旋转手与镊子，修复重置朝向和掌面法向，调整被动弹簧，重新训练带朝向成功条件的 PPO。'),
 ('s4','继续完成朝向修正版及多视角媒体交付；独立测试 63/64。'),
 ('s5','当时按持续四角色支撑实现，并加入弯尾工具；BC + PPO 独立严格成功 9/64。该解释后来被你明确纠正。'),
 ('s6','开始取消持续支撑、改成相对宽面夹持；10:07 收到停止指令，后续由补充要求接续。'),
 ('s6','进一步明确直尾形态；制作 180 mm 直尾工具，验证宽面朝内压力和尖隙响应，再做 BC + PPO。'),
 ('s6','10:16 错误卡显示用量限制。恢复后接续已有姿态与物理试验，完成新策略及视频，独立测试 50/64。'),
 ('s7','把中指放到下方留隙待命，允许约 12°虎口倾斜；以匹配松动对照验证被动承托，另训练 PPO，独立测试 61/64。'),
 ('s8','增加目标条件、完整放置与松开撤离，采用分阶段控制 + 残差 PPO；28×30 cm 随机工作区，独立 64/64，边界 4/4。'),
 ('archive','本次把真实请求、技术路线、结果、失败原因、图片及完整示范视频整理为一个离线 HTML 文件。')
]

audit_rows=[];image_assets=[]
def image_block(item):
 p=ROOT/item['path'];assert p.exists()
 # Report-only encoding: no generated pixels or changed scene content.
 im=Image.open(p).convert('RGB');im.thumbnail((1900,1900))
 b=io.BytesIO();im.save(b,format='JPEG',quality=92,optimize=True)
 data='data:image/jpeg;base64,'+base64.b64encode(b.getvalue()).decode()
 image_assets.append(dict(source=item['path'],source_sha256=digest(p),embedded_bytes=len(b.getvalue()),operation='JPEG compression / resizing only'))
 return f'<figure class="image"><button class="zoom" type="button" aria-label="放大：{esc(item["title"])}"><img loading="lazy" src="{data}" alt="{esc(item["title"])}"></button><figcaption><strong>{esc(item["title"])}</strong><span>{esc(item["caption"])}</span></figcaption></figure>'
def video_block(item):
 d=media[item['id']];p=ROOT/d['report_copy'];assert digest(p)==d['report_sha256']
 return f'''<figure class="film"><video id="video-{d['id']}" controls playsinline preload="none" poster="{uri(ROOT/d['poster'],'image/jpeg')}"><source src="{uri(p,'video/mp4')}" type="video/mp4">请在浏览器中打开本文件以播放内嵌视频。</video><figcaption><div class="film-label"><strong>{esc(item['title'])}</strong><span>{d['duration_s']:.1f}s</span></div><span>{esc(item['caption'])}</span></figcaption></figure>'''
def sources(paths):
 items=[]
 for name in paths:
  p=ROOT/name;assert p.exists(),name
  d=dict(path=name,sha256=digest(p));audit_rows.append(d)
  items.append(f'<li><code>{esc(name)}</code><small>SHA-256 · {d["sha256"][:20]}…</small></li>')
 return '<details class="evidence"><summary>核对依据 · '+str(len(items))+' 项</summary><ul>'+''.join(items)+'</ul></details>'

css='''
@font-face{font-family:ReportCJK;src:url(__FONT__) format('woff');font-display:swap}
:root{--ink:#192c3b;--muted:#586a76;--line:#dce4e7;--paper:#f4f6f5;--teal:#087f81;--orange:#b65c28;--navy:#122f41}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:85px}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.85 ReportCJK,system-ui,sans-serif}a{color:var(--teal);text-decoration:none}button,select{font:inherit}button{cursor:pointer}code{font-family:ui-monospace,monospace;overflow-wrap:anywhere;font-size:.85em}header{background:var(--navy);color:#fff;padding:70px max(5vw,24px) 42px;position:relative;overflow:hidden}header:after{content:'';position:absolute;width:360px;height:360px;border:1px solid #4b7280;border-radius:50%;right:-180px;top:-140px;pointer-events:none}.eyebrow{letter-spacing:.16em;font-size:12px;color:#8bd6d1;text-transform:uppercase}h1{font-size:clamp(30px,4.5vw,56px);line-height:1.3;margin:18px 0}header p{max-width:820px;color:#cfdee6;margin:0 0 24px}.hero-meta{display:flex;gap:10px;flex-wrap:wrap}.hero-meta span{border:1px solid #56737f;border-radius:30px;padding:4px 14px;font-size:12px}.topnav{position:sticky;top:0;z-index:8;background:#fffffff2;backdrop-filter:blur(12px);border-bottom:1px solid var(--line);display:flex;gap:22px;padding:13px max(5vw,24px);overflow:auto;white-space:nowrap;font-size:13px}.topnav a{color:var(--ink)}.topnav a:hover{color:var(--teal)}main{max-width:1210px;margin:auto;padding:36px 30px 80px}h2{font-size:30px;line-height:1.45;margin:0 0 16px}h3{font-size:19px;line-height:1.6;margin:25px 0 9px}p{margin:10px 0 16px}.lead{font-size:19px;max-width:950px}.note{padding:17px 22px;border-left:3px solid var(--teal);background:#e9f3f2;border-radius:0 8px 8px 0;font-size:14px}.note.warm{background:#fcf2e8;border-color:#bd7138}.overview-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:26px 0}.overview-grid article{border:1px solid var(--line);padding:20px;background:white;border-radius:12px}.value{font-size:31px;font-weight:bold;color:var(--teal);line-height:1.4}.sub{font-size:12px;color:var(--muted)}.stage{background:#fff;border:1px solid var(--line);border-radius:16px;padding:30px;margin:32px 0;box-shadow:0 4px 15px #142e4004}.stage-head{display:flex;gap:22px;align-items:flex-start}.number{font-size:38px;line-height:1.3;font-weight:bold;color:#a2bdbd;flex-shrink:0}.date{font-size:12px;color:var(--muted);letter-spacing:.04em}.tag{display:inline-block;color:var(--teal);background:#e9f4f2;border-radius:5px;padding:3px 9px;font-size:12px;margin-bottom:12px}.request{padding:18px 22px;background:#f4f6f7;border-radius:8px;margin:16px 0}.request b{display:block;color:var(--muted);font-size:12px;letter-spacing:.1em;margin-bottom:5px}.request p{margin:0;font-size:17px}.columns{display:grid;grid-template-columns:1.15fr 1fr;gap:32px}.columns ul{margin:0;padding-left:22px}.columns li{margin-bottom:9px}.result{padding:1px 0}.lesson{font-size:14px;color:#705334;border-top:1px solid #e5dacc;padding-top:12px}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:15px;border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:18px 0;margin:24px 0}.metrics b{font-size:24px;color:var(--teal);display:block}.metrics span{font-size:12px;color:var(--muted)}.gallery{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px;margin:22px 0}.gallery.single{grid-template-columns:1fr}.gallery figure{margin:0;border:1px solid var(--line);border-radius:10px;overflow:hidden;background:#fff;min-width:0}video,img{display:block;width:100%;height:auto}video{background:#e2e7e9}.zoom{display:block;border:0;padding:0;background:#fff;width:100%;cursor:zoom-in}.zoom img{max-height:640px;object-fit:contain}.film-label{display:flex;justify-content:space-between;gap:15px}.film-label>span{font-size:12px;white-space:nowrap;color:var(--teal)}figcaption{padding:14px 17px;font-size:13px;line-height:1.75}figcaption strong{display:block;font-size:15px}figcaption>span{display:block;color:var(--muted);margin-top:5px}details{border-top:1px solid var(--line);margin-top:16px;padding-top:12px}summary{cursor:pointer;color:var(--teal);font-size:14px}details ul{padding-left:21px}.evidence li{margin:10px 0;font-size:13px}.evidence small{display:block;color:#6f7e85;font-size:11px}.table-wrap{overflow:auto;background:white;border:1px solid var(--line);border-radius:10px}table{width:100%;border-collapse:collapse;font-size:13px;min-width:800px}th,td{text-align:left;vertical-align:top;border-bottom:1px solid var(--line);padding:14px}th{background:#eaf1f2}td:first-child{font-weight:bold}tbody tr:last-child td{border:0}.flow{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:24px 0}.flow span{border:1px solid var(--line);border-radius:8px;padding:10px 15px;background:#fff;font-size:13px}.flow i{font-style:normal;color:var(--teal)}.plot-layout{display:grid;grid-template-columns:1.3fr 1fr;gap:24px;align-items:center;background:#f6f8f8;padding:24px;border-radius:10px}.plot-layout svg{width:100%;height:auto}.plot-layout select{max-width:100%;padding:9px;border-radius:6px;border:1px solid var(--line);background:white;font-size:13px}.plot-values{font-size:14px;line-height:2}.legend{font-size:12px;color:var(--muted)}.round{padding:22px 0;border-bottom:1px solid var(--line)}.round-head{font-size:13px;display:flex;justify-content:space-between;gap:15px;color:var(--muted)}blockquote{margin:12px 0;padding:12px 20px;border-left:3px solid #b8cbd0;background:#f5f8f9;white-space:pre-wrap;font-size:14px;overflow-wrap:anywhere}.response{font-size:14px}.raw{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px;background:#f4f6f7;border-radius:6px;padding:16px;max-height:450px;overflow:auto}.archive-tools{display:flex;gap:12px;flex-wrap:wrap}.archive-tools button,.print{border:1px solid var(--line);border-radius:7px;padding:8px 15px;background:white;color:var(--ink);font-size:13px}.interrupt{display:grid;grid-template-columns:145px 1fr;gap:20px;border-bottom:1px solid var(--line);padding:17px 0;font-size:14px}.interrupt time{color:var(--muted)}footer{border-top:1px solid var(--line);padding:28px 0;font-size:12px;color:var(--muted)}dialog{border:0;border-radius:12px;max-width:95vw;max-height:95vh;padding:16px;background:#fff}dialog::backdrop{background:#102d41d9}dialog img{max-height:83vh;max-width:90vw;width:auto;object-fit:contain}dialog button{float:right;border:0;border-radius:4px;padding:7px 12px;background:#e9efef;margin-bottom:8px}.progress{height:3px;position:fixed;top:0;left:0;background:#29aca4;z-index:20;width:0}.closing{padding:25px;background:var(--navy);color:#fff;border-radius:12px;margin-top:30px}.closing p{color:#d4e3e8}
@media(max-width:760px){body{font-size:15px}header{padding-top:40px}main{padding:22px 16px}.overview-grid{grid-template-columns:repeat(2,1fr)}.stage{padding:20px 17px}.stage-head{gap:12px}.number{font-size:28px}h2{font-size:24px}.columns,.gallery,.plot-layout{grid-template-columns:1fr}.metrics{gap:10px}.metrics b{font-size:19px}.metrics span{font-size:11px}.topnav{gap:18px;padding:13px 18px}.interrupt{grid-template-columns:1fr;gap:4px}.value{font-size:26px}.plot-layout{padding:12px}header:after{display:none}}
@media print{.topnav,.progress,.archive-tools,.print,dialog{display:none!important}body{background:#fff;font-size:10pt}header{background:#fff;color:#152f42;padding:10px}header p{color:#555}main{padding:0;max-width:none}.stage{box-shadow:none;break-inside:auto;border-radius:0}.gallery figure,.metrics,.round{break-inside:avoid}video{max-height:230px}.evidence{display:none}h2{font-size:20pt}.columns{display:block}details:not([open])>*:not(summary){display:none}}
'''

chunks=['''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="description" content="UR5e + XHand 使用镊子：逐轮用户反馈、技术演进、验证结果及内嵌仿真视频。"><title>tweezers｜对话与实验演进记录</title><style>''',css,'''</style></head><body><div class="progress" id="progress"></div>
<header><div class="eyebrow">UR5e × XHand / Experiment journal</div><h1>从夹起方块<br>到随机位置完整抓放</h1><p>一次围绕真实握姿、接触机制与任务泛化的迭代记录。逐轮保留你的要求，说明实际采用的技术路线、取得的结果，以及哪些判断随后被修正。</p><div class="hero-meta"><span>2026.10.15 — 2026.10.19</span><span>18 条业务请求</span><span>8 个主要阶段</span><span>10 段内嵌视频</span><span>单文件 · 可离线查看</span></div></header>
<nav class="topnav" aria-label="报告导航"><a href="#overview">全程概览</a><a href="#s1">控制基线</a><a href="#s2">人类参考</a><a href="#s4">朝向修正</a><a href="#s6">宽面夹持</a><a href="#s7">备用承托</a><a href="#s8">随机抓放</a><a href="#rounds">逐轮原话</a><a href="#interruptions">中断记录</a></nav><main>
<section id="overview"><div class="tag">截至本次整理，已交付的最终状态</div><h2>要求如何一步步变成可验证的行为</h2><p class="lead">最终使用直尾镊子：拇指、食指从相对宽面挤压；中指在下方备用，虎口允许倾斜。在限定桌面工作区内，完成夹起、搬运、放下、松开与撤离。</p>
<div class="overview-grid"><article><div class="value">28 × 30 cm</div><div class="sub">最终连续随机工作区</div></article><article><div class="value">64 / 64</div><div class="sub">最终独立随机测试，本批结果</div></article><article><div class="value">3.384 mm</div><div class="sub">该批最大水平落点误差</div></article><article><div class="value">4 / 4</div><div class="sub">约 41 cm 的边界对角搬运</div></article></div>
<div class="note">本报告按飞书实际消息及项目保存证据重建。主文是核对后的技术回顾；“逐轮原话”保留业务请求原文，附录保留阶段答复。重复进度卡、工具调用流水、状态/模型切换命令不逐条展开。时间沿用飞书返回的显示时间；随机种子如 <code>20261021</code> 是整数标识，不是实验日期。</div>
<h3>技术路线的变化</h3><div class="flow"><span>状态机 + IK 基线</span><i>→</i><span>二维人类握姿先验 + PPO</span><i>→</i><span>姿态与接触角色约束</span><i>→</i><span>物理演示 BC + PPO</span><i>→</i><span>目标条件残差 PPO + 阶段控制</span></div>
<div class="table-wrap"><table><thead><tr><th>阶段</th><th>实际方法</th><th>结果</th><th>测试范围 / 不可混用之处</th></tr></thead><tbody>
<tr><td>控制基线</td><td>状态机、IK、位姿反馈；无 RL</td><td>6 个确定性案例通过</td><td>不是随机成功率；初始化有 0.6 秒支撑。</td></tr>
<tr><td>早期人类参考 PPO</td><td>69 状态 / 4 动作</td><td>48/64；接近版 58/64</td><td>两版起点、姿态和接触条件不同；XY ±2 mm。</td></tr>
<tr><td>掌面/工具朝向修正</td><td>从随机权重训练 PPO</td><td>63/64 独立测试</td><td>新增朝向门槛；预持镊，XY ±2 mm。</td></tr>
<tr><td>持续三指/虎口支撑</td><td>物理演示 BC + PPO</td><td>9/64 严格独立测试</td><td>错误地要求持续支撑，使用弯尾；后续已撤销。</td></tr>
<tr><td>直尾、相对宽面</td><td>BC + PPO，90 状态 / 6 动作</td><td>50/64 独立测试</td><td>取消持续支撑；尚未专项证明备用承托。</td></tr>
<tr><td>中指下方备用 · v14</td><td>BC + PPO；另做被动承托对照</td><td>61/64 独立测试</td><td>局部夹取；承托试验与 RL 评估是两组不同实验。</td></tr>
<tr><td>随机抓放 · v15</td><td>几何阶段控制 + 目标条件残差 PPO</td><td>64/64；边界另 4/4</td><td>28×30 cm 工作区，新增放下、松开与撤离验收。</td></tr>
</tbody></table></div><p class="sub">各行的初态、成功条件和任务范围改变过，因此不把它们画成同一基准上的“成功率持续提升”曲线。</p></section>
''']

holdout=read('runs/pick_place15_ppo101_holdout20261017/evaluation.json')
for i,s in enumerate(stages,1):
 chunks.append(f'<section class="stage" id="{s["id"]}"><div class="stage-head"><div class="number">{i:02d}</div><div><div class="date">2026 年 · {esc(s["date"])}</div><h2>{esc(s["title"])}</h2><span class="tag">{esc(s["tag"])}</span></div></div>')
 chunks.append(f'<div class="request"><b>你的要求</b><p>{esc(s["request"])}</p></div><div class="columns"><div><h3>采用的技术路线</h3><ul>'+''.join(f'<li>{esc(t)}</li>' for t in s['route'])+'</ul></div>')
 chunks.append(f'<div class="result"><h3>取得的结果</h3><p>{esc(s["result"])}</p><p class="lesson"><b>当时的边界与后续修正</b><br>{esc(s["lesson"])}</p></div></div>')
 chunks.append('<div class="metrics">'+''.join(f'<div><b>{esc(v)}</b><span>{esc(t)}</span></div>' for v,t in s['metrics'])+'</div>')
 if s['id']=='s7':
  chunks.append('''<div class="note">同一松动控制下，仅改变中指位置：下方备用时，后半段平均工具重心下降 <b>1.87 mm</b>；中指收回时为 <b>78.89 mm</b>。这项对照验证“必要时被动托住”，没有把它归功于强化学习主动恢复。</div>''')
 if s['videos']:chunks.append('<div class="gallery '+('single' if len(s['videos'])==1 else '')+'">'+''.join(video_block(x) for x in s['videos'])+'</div>')
 if s['images']:chunks.append('<div class="gallery '+('single' if len(s['images'])%2 else '')+'">'+''.join(image_block(x) for x in s['images'])+'</div>')
 if s['id']=='s8':
  chunks.append('''<h3>独立随机测试：实际采样点</h3><div class="plot-layout"><div><svg id="workspace" viewBox="0 0 530 490" role="img" aria-label="64次独立测试的实际起点和目标坐标"></svg><div class="legend">蓝色空心圆：起点 · 橙色小圆：目标 · 实线：当前选中样本的起终点连线（不表示实际运动轨迹）</div></div><div><label for="case-select">查看某次独立测试</label><br><select id="case-select">'''+''.join(f'<option value="{n}">样本 {n+1:02d} · 误差 {e*1000:.3f} mm</option>' for n,e in enumerate(holdout['final_xy_error_m']))+'''</select><div class="plot-values" id="case-values"></div><p class="sub">数据来自冻结检查点后的 64 回合原始评估。该批实际搬运距离约 12.02—30.49 cm；约 41 cm 的最大跨度由上方四组边界测试另外覆盖。</p></div></div><h3>验收实际检查了什么</h3><p>双颚夹住并抬升至少 45 mm、持续 0.25 秒；落点误差小于 10 mm、落桌高度误差小于 1.5 mm、方块静稳；双颚不再接触方块，镊尖撤离至少 35 mm，工具仍被正确保持；最终条件持续 0.75 秒。</p><p class="sub">独立重构覆盖 64 回合、77,711 个有效控制样本；关节限位与机器人桌面间隙均通过。另对近景录像的 11,480 个物理子步检查手部源表面相交、关节限位、朝向和桌面间隙。工具允许触桌，0.1 mm 是穿入检查容差，不能改写为“工具绝不接触桌面”。</p>''')
 chunks.append(sources(s['evidence'])+'</section>')

chunks.append('''<section class="stage" id="rounds"><span class="tag">可追溯的需求链</span><h2>逐轮原话 → 实际回应与结果</h2><p>以下保留这段任务中的全部 18 条业务请求，含进度询问、重复澄清和继续执行。每条后面的回应摘要为本次结合项目证据所作的整理，不伪装成当时的逐字回复。</p>''')
for n,(m,(stage,answer)) in enumerate(zip(users,annotations),1):
 chunks.append(f'<article class="round" id="round-{n}"><div class="round-head"><b>第 {n:02d} 条 · 你的原话</b><time>{esc(m["create_time"])}</time></div><blockquote>{esc(m["content"])}</blockquote><div class="response"><b>回应 / 落实结果：</b>{esc(answer)}'+(f' <a href="#{stage}">查看对应阶段 ↑</a>' if stage!='archive' else '')+'</div></article>')
chunks.append('</section>')

chunks.append('''<section class="stage" id="interruptions"><span class="tag">执行过程记录</span><h2>中断与停顿，不与算法失败混为一谈</h2>
<div class="interrupt"><time>10-15 · 17:37 / 18:03</time><div>错误消息明确包含 <code>remote compact task</code>、<code>stream disconnected</code> 和 <code>Transport error: timeout</code>。对应远程上下文压缩/连接中断，不能据此推断训练已经完成。20:24 恢复时说明视频与 7 帧姿态已准备，但 PPO 尚未开始。</div></div>
<div class="interrupt"><time>10-15 · 17:56</time><div>收到你的 <code>/stop</code>，会话显示“已被中断”；之后的继续请求恢复原任务。</div></div>
<div class="interrupt"><time>10-17 · 10:07</time><div>收到第二次 <code>/stop</code>。10:08 的补充请求加入“不要弯尾”的要求，因此最终实现以补充后的要求为准。</div></div>
<div class="interrupt"><time>10-17 · 10:16</time><div>错误卡显示用量限制（<code>usage limit</code>）。10:29 收到继续请求后接续保存的姿态与实验，而非把未完成工作宣称已经完成。</div></div>
<div class="interrupt"><time>实际研发瓶颈</time><div>早期下载和依赖安装耗时；自由持镊会滑落；掌面旋转后原接触布局失稳；持续支撑门槛导致较低成功率；中指位置改变了镊尖高度差；随机搬运阶段出现近基座 IK 失败及释放后空镊滑脱。这些有具体物理/算法原因，分别通过接触调整、约束澄清、重训、工作区外移和握持恢复解决。</div></div>
<p class="sub">中断时间来自本次取回的飞书错误卡与停止指令。没有从耗时推测 GPU 守护程序故障，也没有把全部等待归咎于算力。</p></section>
<section class="stage" id="archive"><h2>阶段答复与证据索引</h2><p>保留飞书中的阶段性文字答复供核对。重复进度卡与工具流水不展开，主文对其最终证据进行了归纳。</p><div class="archive-tools"><button id="toggle-replies" type="button">展开 / 收起全部答复</button><button id="download-evidence" type="button">导出本报告证据索引 JSON</button><button type="button" class="print" onclick="window.print()">打印 / 保存 PDF</button></div>''')
for m in replies:
 chunks.append(f'<details class="reply"><summary>{esc(m["create_time"])} · 阶段文字答复</summary><div class="raw">{esc(m["content"])}</div></details>')
chunks.append('''<div class="closing"><h3>最后保留下来的边界</h3><p>任务始终从已持镊状态开始；没有完成从桌面获取工具、视觉定位或实机迁移。人类视频提供的是二维弱姿态先验与握持布局参考。最终随机抓放使用仿真状态、几何阶段控制和残差 PPO；其可靠性证据限定在已测试工作区与物体条件内。</p></div></section>
<footer><b>文档构建：2026-10-19</b><p>所有展示图片均来自实际参考帧、项目分析图或仿真输出；视频由已交付原片压缩，不增删帧、不修改物体动作、保留原片播放速度。报告不重新训练策略，也不把历史失败分支写成当前成果。视频、图片、字体和交互脚本均内嵌；下载后用 Chrome、Edge 或其他支持 H.264 的浏览器打开。飞书附件预览若不播放视频，请下载文件后打开。</p></footer></main>
<dialog id="lightbox"><button type="button" aria-label="关闭大图">关闭 ×</button><img alt="放大图"></dialog>''')

index=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='18 substantive user requests, 8 development stages, selected direct assistant replies and 10 complete demonstration videos',
 source_chat_export_sha256=digest(ROOT/'reports/history_lark_messages_20261019.json'),
 user_requests=[dict(ordinal=n,time=m['create_time'],text=m['content'],stage=annotations[n-1][0]) for n,m in enumerate(users,1)],
 source_reports=list({x['path']:x for x in audit_rows}.values()),videos=list(media.values()),images=image_assets,
 notes=['Source message timestamps retained as returned by Lark.', 'Integer evaluation seeds are not dates.', 'Video report copies preserve full frame sequences and original playback speed.', 'Per-version success definitions differ; no unified cross-version benchmark is claimed.'])
index_json=json.dumps(index,ensure_ascii=False).replace('</','<\\/')
plot_json=json.dumps(dict(start=holdout['start'],goal=holdout['goal'],error=holdout['final_xy_error_m'],success=holdout['success_per_env']))
chunks.append('<script type="application/json" id="evidence-data">'+index_json+'</script><script type="application/json" id="plot-data">'+plot_json+'</script>')
chunks.append('''<script>
const videos=[...document.querySelectorAll('video')];videos.forEach(v=>v.addEventListener('play',()=>videos.forEach(o=>{if(o!==v)o.pause()})));
const box=document.querySelector('#lightbox');document.querySelectorAll('.zoom').forEach(b=>b.addEventListener('click',()=>{box.querySelector('img').src=b.querySelector('img').src;box.querySelector('img').alt=b.querySelector('img').alt;box.showModal()}));box.querySelector('button').onclick=()=>box.close();box.addEventListener('click',e=>{if(e.target===box)box.close()});
document.querySelector('#toggle-replies').onclick=()=>{const d=[...document.querySelectorAll('details.reply')],open=d.some(x=>!x.open);d.forEach(x=>x.open=open)};
document.querySelector('#download-evidence').onclick=()=>{const blob=new Blob([JSON.stringify(JSON.parse(document.querySelector('#evidence-data').textContent),null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='tweezers_history_evidence.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
addEventListener('scroll',()=>{let n=document.documentElement.scrollHeight-innerHeight;document.querySelector('#progress').style.width=(n?scrollY/n*100:0)+'%'},{passive:true});
const data=JSON.parse(document.querySelector('#plot-data').textContent),svg=document.querySelector('#workspace'),NS='http://www.w3.org/2000/svg';
function el(tag,attrs,parent=svg){let e=document.createElementNS(NS,tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));parent.appendChild(e);return e}
const X=x=>65+(x-.48)/.28*392,Y=y=>425-(y+.22)/.30*420;
el('rect',{x:65,y:5,width:392,height:420,fill:'#fff',stroke:'#b8c9cd'});
for(let x=.48;x<=.761;x+=.07){let xx=X(x);el('line',{x1:xx,y1:5,x2:xx,y2:425,stroke:'#e2eaed'});let t=el('text',{x:xx,y:451,'text-anchor':'middle',fill:'#627782','font-size':12});t.textContent=x.toFixed(2)}
for(let y=-.22;y<=.081;y+=.06){let yy=Y(y);el('line',{x1:65,y1:yy,x2:457,y2:yy,stroke:'#e2eaed'});let t=el('text',{x:53,y:yy+4,'text-anchor':'end',fill:'#627782','font-size':12});t.textContent=y.toFixed(2)}
el('text',{x:260,y:480,'text-anchor':'middle','font-size':13,fill:'#355361'}).textContent='X / m';el('text',{x:10,y:230,transform:'rotate(-90 10 230)','text-anchor':'middle','font-size':13,fill:'#355361'}).textContent='Y / m';
data.start.forEach((s,i)=>{el('circle',{cx:X(s[0]),cy:Y(s[1]),r:3.3,stroke:'#2881b1',fill:'none',opacity:.55});el('circle',{cx:X(data.goal[i][0]),cy:Y(data.goal[i][1]),r:2.6,fill:'#c3743a',opacity:.5})});
let selected=el('g',{});function show(i){selected.replaceChildren();let s=data.start[i],g=data.goal[i];el('line',{x1:X(s[0]),y1:Y(s[1]),x2:X(g[0]),y2:Y(g[1]),stroke:'#152f42','stroke-width':2.2},selected);el('circle',{cx:X(s[0]),cy:Y(s[1]),r:7,stroke:'#187bb0','stroke-width':3,fill:'white'},selected);el('circle',{cx:X(g[0]),cy:Y(g[1]),r:7,fill:'#ce6d26',stroke:'white','stroke-width':2},selected);document.querySelector('#case-values').innerHTML='<h3>样本 '+String(i+1).padStart(2,'0')+' · '+(data.success[i]?'通过':'未通过')+'</h3>起点：('+s[0].toFixed(4)+', '+s[1].toFixed(4)+') m<br>目标：('+g[0].toFixed(4)+', '+g[1].toFixed(4)+') m<br>搬运距离：'+(100*Math.hypot(s[0]-g[0],s[1]-g[1])).toFixed(2)+' cm<br>最终水平误差：<b>'+(data.error[i]*1000).toFixed(3)+' mm</b>'}
document.querySelector('#case-select').onchange=e=>show(Number(e.target.value));show(0);
</script></body></html>''')
document=''.join(chunks)
# Embed only the glyphs used by the document, including interactive labels.
plain=re.sub(r'data:[^;]+;base64,[A-Za-z0-9+/=]+','',document)
plain=re.sub(r'<[^>]+>',' ',plain)
font=TTFont(ROOT/'assets/fonts/NotoSansCJKsc-Regular.otf')
options=subset.Options();options.flavor='woff';options.layout_features=['*']
sub=subset.Subsetter(options=options);sub.populate(text=html.unescape(plain));sub.subset(font)
font.flavor='woff';bio=io.BytesIO();font.save(bio)
document=document.replace('__FONT__','data:font/woff;base64,'+base64.b64encode(bio.getvalue()).decode())
OUT.write_text(document,encoding='utf-8')
index['output']=str(OUT.relative_to(ROOT));index['output_bytes']=OUT.stat().st_size;index['output_sha256']=digest(OUT);index['embedded_font_bytes']=len(bio.getvalue())
(ROOT/'reports/conversation_history_manifest.json').write_text(json.dumps(index,ensure_ascii=False,indent=2))
print(json.dumps(dict(html=str(OUT),bytes=OUT.stat().st_size,stages=len(stages),requests=len(users),replies=len(replies),videos=len(media),images=len(image_assets),font_bytes=len(bio.getvalue())),ensure_ascii=False,indent=2))
