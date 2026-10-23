"""Check the offline report in a real browser, including all embedded videos."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]
HTML=ROOT/'outputs/tweezers_对话与实验演进记录_20261019.html'
OUT=ROOT/'reports/conversation_history_preview';OUT.mkdir(exist_ok=True)
report=dict(page_errors=[],external_requests=[],videos=[],images=[])
with sync_playwright() as p:
 browser_path=Path('/tmp/history-browser-parts/installed/chrome-headless-shell-linux64/chrome-headless-shell')
 launch={'executable_path':str(browser_path)} if browser_path.exists() else {}
 browser=p.chromium.launch(headless=True,args=['--no-sandbox','--disable-dev-shm-usage'],**launch)
 page=browser.new_page(viewport=dict(width=1440,height=1050),device_scale_factor=1)
 page.on('pageerror',lambda error:report['page_errors'].append(str(error)))
 def request(req):
  if req.url.startswith(('http:','https:')):report['external_requests'].append(req.url)
 page.on('request',request)
 page.goto(HTML.as_uri(),wait_until='load',timeout=60000)
 page.evaluate('document.fonts.ready')
 report['font_loaded']=page.evaluate("document.fonts.check('16px ReportCJK')")
 page.screenshot(path=str(OUT/'desktop_top.png'))
 assert page.locator('article.round').count()==18
 assert page.locator('video').count()==10
 for video in page.locator('video').all():
  result=video.evaluate('''v=>new Promise(async(resolve,reject)=>{
   const timer=setTimeout(()=>reject(new Error('video timed out '+v.id)),20000);
   try{v.muted=true;v.preload='auto';v.load();await new Promise((ok,no)=>{v.onloadeddata=ok;v.onerror=()=>no(new Error('media error '+v.error?.code))});
   await v.play();await new Promise(ok=>setTimeout(ok,350));v.pause();
   const result={id:v.id,duration:v.duration,width:v.videoWidth,height:v.videoHeight,currentTime:v.currentTime,error:v.error?.code||null};
   v.currentTime=Math.max(0,v.duration-.3);await new Promise((ok,no)=>{v.onseeked=ok;v.onerror=no});
   result.endSeek=v.currentTime;clearTimeout(timer);resolve(result)}catch(e){clearTimeout(timer);reject(e)}
  })''')
  assert result['currentTime']>0 and result['width']>0 and result['error'] is None,result
  report['videos'].append(result)
 for img in page.locator('figure.image img').all():
  img.scroll_into_view_if_needed();img.evaluate('i=>i.decode()')
  report['images'].append(img.evaluate('i=>({width:i.naturalWidth,height:i.naturalHeight,alt:i.alt})'))
 page.locator('#s7').scroll_into_view_if_needed()
 page.screenshot(path=str(OUT/'middle_backup_section.png'))
 page.locator('#case-select').select_option('63')
 assert '样本 64' in page.locator('#case-values').inner_text()
 report['interactive_sample_selector']=True
 page.locator('#workspace').scroll_into_view_if_needed()
 page.screenshot(path=str(OUT/'random_workspace.png'))
 page.locator('.zoom').first.click()
 assert page.locator('#lightbox').evaluate('d=>d.open')
 page.locator('#lightbox button').click();report['image_zoom']=True
 page.locator('#toggle-replies').click()
 assert page.locator('details.reply[open]').count()==13
 page.locator('#toggle-replies').click();report['reply_toggle']=True
 with page.expect_download() as download:
  page.locator('#download-evidence').click()
 path=download.value.path();data=json.loads(Path(path).read_text())
 assert len(data['user_requests'])==18
 report['evidence_download']=True
 report['desktop_overflow']=page.evaluate('document.documentElement.scrollWidth>innerWidth')
 page.set_viewport_size(dict(width=390,height=844));page.goto(HTML.as_uri(),wait_until='load');page.evaluate('document.fonts.ready')
 page.screenshot(path=str(OUT/'mobile_top.png'))
 report['mobile_overflow']=page.evaluate('document.documentElement.scrollWidth>innerWidth')
 page.locator('#s8').scroll_into_view_if_needed();page.screenshot(path=str(OUT/'mobile_final_stage.png'))
 report['passed']=not (report['page_errors'] or report['external_requests'] or report['desktop_overflow'] or report['mobile_overflow']) and report['font_loaded']
 browser.close()
(ROOT/'reports/conversation_history_browser_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False,indent=2))
assert report['passed']
