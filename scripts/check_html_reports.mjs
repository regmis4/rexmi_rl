import { chromium } from '/home/susan/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs';
const browser=await chromium.launch({executablePath:'/usr/bin/google-chrome',headless:true,args:['--no-sandbox']});
for(const name of ['rexmi-hardware-report','robstride-motor-selection']){
  const page=await browser.newPage();
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto('file:///home/susan/rexmi_rl/docs/reports/'+name+'.html');
  for(const width of [1440,390]){
    await page.setViewportSize({width,height:1000});
    const result=await page.evaluate(()=>({overflow:document.documentElement.scrollWidth>innerWidth,broken:[...document.images].filter(i=>!i.closest('dialog')&&(!i.complete||!i.naturalWidth)).length}));
    if(result.overflow||result.broken||errors.length)throw Error(JSON.stringify({name,width,result,errors}));
    await page.screenshot({path:'/tmp/'+name+'-'+width+'.png',fullPage:true});
  }
  await page.locator('figure img').first().click();
  if(!await page.locator('dialog').evaluate(d=>d.open))throw Error('Zoom failed');
  await page.keyboard.press('Escape');
  console.log(name+': desktop/mobile, images, scripts and zoom passed');
  await page.close();
}
await browser.close();
