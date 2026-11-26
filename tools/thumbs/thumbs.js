// render one still per exercise from the app's own 3D view → OUT/<id>.png
// usage: node thumbs.js [limit] [startIndex]
const {chromium}=require(process.env.PW);const fs=require('fs');
const OUT=process.env.OUT;const LIM=+(process.argv[2]||99999),START=+(process.argv[3]||0);
(async()=>{const b=await chromium.launch({args:['--use-gl=swiftshader','--enable-webgl','--ignore-gpu-blocklist']});
const p=await b.newPage({viewport:{width:420,height:900},deviceScaleFactor:1.5});const errs=[];p.on('pageerror',e=>errs.push(e.message));
await p.route('**/three.min.js',r=>r.fulfill({path:process.env.S+'/three/node_modules/three/build/three.min.js',contentType:'application/javascript'}));
await p.route(/gstatic|firebase/,r=>r.abort());
await p.goto('http://localhost:8765/coach-builder.html');await p.waitForTimeout(1500);
const ids=await p.evaluate(()=>EX.filter(e=>!e.own).map(e=>e.id));
let n=0,skip=0,fail=[];const t0=Date.now();
for(const id of ids.slice(START,START+LIM)){
  if(fs.existsSync(OUT+'/'+id+'.png')){skip++;continue;}
  try{
    const has=await p.evaluate(id=>{closeModal();openInfo(id);return !!document.getElementById('tdhost');},id);
    if(!has){fail.push(id+':noframes');continue;}
    await p.waitForFunction(()=>TD.dbg.cur&&document.querySelector('#mbox canvas'),null,{timeout:30000});
    await p.waitForTimeout(150);
    const d=await p.evaluate(()=>TD.dbg.cur.shot(1.6,null,null));
    fs.writeFileSync(OUT+'/'+id+'.png',Buffer.from(d.split(',')[1],'base64'));n++;
  }catch(e){fail.push(id+':'+e.message.slice(0,60));}
}
await p.evaluate(()=>closeModal());
console.log('done',n,'skipped',skip,'failed',fail.length,'secs',Math.round((Date.now()-t0)/1000));if(fail.length)console.log(fail.join('\n'));if(errs.length)console.log(errs.slice(0,5));
await b.close();})();
