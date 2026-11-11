"""Build training_monitor.html: one self-contained page that shows how the
network's parameters change after each image it is fed."""
import json

TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Training Monitor</title>
<style>
/* Layout: a lab notebook page. Run summary, then a scrubber that drives every picture, then the history charts. */
:root{
  --page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);
  --s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300;--s7:#6250d6;--s8:#e34948;
  --neg:#2a78d6;--mid:#f0efec;--pos:#e34948;--warn:#fab219;
  --font:system-ui,-apple-system,"Segoe UI",sans-serif;--mono:ui-monospace,"Cascadia Mono",Consolas,monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);
  --s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;--s7:#9085e9;--s8:#e66767;--neg:#3987e5;--mid:#383835;--pos:#e66767;color-scheme:dark}}
:root[data-theme="dark"]{--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);
  --s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;--s7:#9085e9;--s8:#e66767;--neg:#3987e5;--mid:#383835;--pos:#e66767;color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:14px/1.45 var(--font)}
main{max-width:1180px;margin:0 auto;padding:20px max(16px,env(safe-area-inset-right,0px)) 40px max(16px,env(safe-area-inset-left,0px));display:grid;gap:16px}
h1{font-size:22px;margin:0;text-wrap:balance}
h2{font-size:15px;margin:0 0 2px}
.sub{color:var(--ink2);margin:2px 0 0}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:8px;padding:14px;min-width:0}
.facts{display:flex;flex-wrap:wrap;gap:6px 22px;color:var(--ink2)}
.facts b{color:var(--ink);font-weight:600;font-variant-numeric:tabular-nums}
.banner{border-left:3px solid var(--warn);padding:8px 12px;background:var(--surface);border-radius:4px}
.scrub{display:flex;flex-wrap:wrap;align-items:center;gap:10px}
.scrub input[type=range]{flex:1 1 260px;min-width:0}
.scrub button,.sel{font:inherit;padding:6px 12px;border:1px solid var(--axis);border-radius:6px;background:var(--surface);color:var(--ink);cursor:pointer}
.where{font-variant-numeric:tabular-nums;color:var(--ink2);min-width:0}
.where b{color:var(--ink)}
.now{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(0,2fr) minmax(0,1fr);gap:16px;margin-top:12px}
.filters{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;max-width:260px}
.filters canvas{width:100%;aspect-ratio:1;image-rendering:pixelated;border-radius:2px;display:block}
.legend-scale{display:flex;align-items:center;gap:6px;color:var(--muted);font-size:12px;margin-top:6px}
.legend-scale i{display:block;width:90px;height:8px;border-radius:2px;background:linear-gradient(90deg,var(--neg),var(--mid),var(--pos))}
.previews{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}
.previews figure{margin:0;min-width:0}
.previews img{width:100%;display:block;border-radius:3px;image-rendering:pixelated}
figcaption{font-size:12px;color:var(--muted);margin-top:3px}
.keys{display:flex;flex-wrap:wrap;gap:4px 12px;font-size:12px;color:var(--ink2);margin-top:6px}
.keys span::before{content:"";display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px;background:var(--c)}
.stats{display:grid;grid-template-columns:auto auto;gap:4px 12px;align-content:start;font-variant-numeric:tabular-nums}
.stats dt{color:var(--ink2)} .stats dd{margin:0;text-align:right;font-weight:600}
.charts{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}
.chart svg{width:100%;height:auto;display:block;overflow:visible}
.chart .note{color:var(--muted);font-size:12px;margin:0 0 6px}
.legend{display:flex;flex-wrap:wrap;gap:4px 12px;font-size:12px;color:var(--ink2);margin-top:4px}
.legend span::before{content:"";display:inline-block;width:12px;height:2px;margin-right:5px;vertical-align:3px;background:var(--c)}
.tip{position:fixed;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--ring);border-radius:6px;padding:6px 8px;font-size:12px;box-shadow:0 2px 8px rgba(0,0,0,.15);font-variant-numeric:tabular-nums;z-index:5}
.tip[hidden]{display:none}
.tip .r{display:flex;gap:6px;align-items:center;white-space:nowrap}
.tip .r i{width:8px;height:8px;border-radius:2px;background:var(--c);display:inline-block}
.heat canvas{width:100%;height:220px;display:block;image-rendering:pixelated;border-radius:3px}
.heatwrap{display:grid;grid-template-columns:auto minmax(0,1fr);gap:6px}
.yax{display:flex;flex-direction:column;justify-content:space-between;font-size:11px;color:var(--muted);font-variant-numeric:tabular-nums;text-align:right;height:220px}
details{color:var(--ink2)} details table{border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:12px}
details td,details th{padding:2px 10px;border-bottom:1px solid var(--grid);text-align:right}
.tablebox{overflow-x:auto;max-height:320px}
button:focus-visible,input:focus-visible,select:focus-visible{outline:2px solid var(--s1);outline-offset:2px}
@media (max-width:860px){.now,.charts{grid-template-columns:1fr}.previews{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style></head><body><main>
<header>
  <h1>Training Monitor</h1>
  <p class="sub" id="sub"></p>
</header>
<div class="banner" id="banner" hidden></div>
<section class="card"><div class="facts" id="facts"></div></section>

<section class="card">
  <div class="scrub">
    <button id="play" type="button">Play</button>
    <input id="k" type="range" min="0" value="0" aria-label="Snapshot">
    <span class="where" id="where"></span>
  </div>
  <div class="now">
    <div>
      <h2>First-layer filters</h2>
      <p class="sub" style="font-size:12px">The 16 3×3 kernels that see the raw image first.</p>
      <div class="filters" id="filters"></div>
      <div class="legend-scale">negative <i></i> positive</div>
    </div>
    <div>
      <h2>What the network sees in a fixed test patch</h2>
      <div class="previews">
        <figure><img id="pv-raw" alt="Raw SEM patch"><figcaption>image</figcaption></figure>
        <figure><img id="pv-target" alt="Marks from folder B"><figcaption>marks (folder B)</figcaption></figure>
        <figure><img id="pv-start" alt="Prediction before training"><figcaption>before training</figcaption></figure>
        <figure><img id="pv-now" alt="Prediction at this snapshot"><figcaption id="pv-now-cap">after this image</figcaption></figure>
      </div>
      <div class="keys"><span style="--c:#282c34">background</span><span style="--c:#2f7fa8">dense</span><span style="--c:#d9771f">cauliflower</span><span style="--c:#6e6e6e">ignored</span></div>
    </div>
    <dl class="stats" id="stats"></dl>
  </div>
</section>

<section class="charts">
  <div class="card chart" id="c-loss"></div>
  <div class="card chart" id="c-drift"></div>
  <div class="card chart" id="c-step"></div>
  <div class="card chart" id="c-grad"></div>
</section>

<section class="card heat">
  <div class="scrub" style="justify-content:space-between">
    <div><h2>Weight distribution over time</h2><p class="sub" style="font-size:12px">Each column is one snapshot; stronger colour means more weights at that value (log scale, so thin tails stay visible).</p></div>
    <label>Block <select class="sel" id="hb"></select></label>
  </div>
  <div class="heatwrap" style="margin-top:8px"><div class="yax" id="hy"></div><canvas id="heat"></canvas></div>
  <details style="margin-top:10px"><summary>Numbers for this block</summary><div class="tablebox"><table id="tbl"></table></div></details>
</section>
</main>
<div class="tip" id="tip" hidden></div>
<script>
const LOG=__LOG__;
const S=LOG.snapshots, B=LOG.blocks, N=S.length;
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const COL=B.map((_,i)=>"var(--s"+(i+1)+")");
const $=id=>document.getElementById(id);
const fmt=(v,d=3)=>v==null?"–":(Math.abs(v)<1e-3&&v!==0?v.toExponential(1):(+v).toFixed(d));
let k=N; // 0 = start, j = after j-th image

$("sub").textContent=`Run of ${LOG.created.replace("T"," ")} · ${LOG.images.length} images × ${LOG.settings.epochs} epochs · ${LOG.settings.steps} steps per image`;
if(LOG.labels!=="human"){$("banner").hidden=false;$("banner").textContent="This run trained on the draft marks from workflow 1 as well as hand labels. It shows the monitor working; the network has only learned to copy the draft rule."}
const last=S[N-1];
$("facts").innerHTML=[["Parameters",LOG.n_params.toLocaleString()],["Images fed",N],["Labeled particles",LOG.images.reduce((a,b)=>a+b.particles,0).toLocaleString()],["Final training loss",fmt(last.loss)],["Loss on unseen image, last epoch",fmt(S.filter(s=>s.epoch===LOG.settings.epochs).reduce((a,s)=>a+s.loss_before,0)/LOG.images.length)]].map(([a,b])=>`<span>${a} <b>${b}</b></span>`).join("");

// filters
const fbox=$("filters");const fc=[];for(let i=0;i<16;i++){const c=document.createElement("canvas");c.width=3;c.height=3;c.title="filter "+(i+1);fbox.append(c);fc.push(c)}
function hex2rgb(h){h=h.replace("#","");if(h.length===3)h=h.split("").map(x=>x+x).join("");return[0,2,4].map(i=>parseInt(h.slice(i,i+2),16))}
function drawFilters(F){const neg=hex2rgb(css("--neg")),mid=hex2rgb(css("--mid")),pos=hex2rgb(css("--pos"));
  const m=Math.max(...S.flatMap(s=>s.filters.flat(2)).map(Math.abs),...LOG.start.filters.flat(2).map(Math.abs));
  F.forEach((f,i)=>{const ctx=fc[i].getContext("2d"),im=ctx.createImageData(3,3);f.flat().forEach((v,j)=>{const t=Math.max(-1,Math.min(1,v/m)),a=t<0?neg:pos,u=Math.abs(t);
    for(let c=0;c<3;c++)im.data[j*4+c]=Math.round(mid[c]+(a[c]-mid[c])*u);im.data[j*4+3]=255});ctx.putImageData(im,0,0)})}
$("pv-raw").src=LOG.preview.raw;$("pv-target").src=LOG.preview.target;$("pv-start").src=LOG.start.preview;

function update(){
  const s=k?S[k-1]:null;
  $("k").value=k;
  $("where").innerHTML=k?`After image <b>${k}</b> of ${N} · epoch ${s.epoch} · <b>${s.image}</b>`:"<b>Start</b>, before any image";
  drawFilters(s?s.filters:LOG.start.filters);
  $("pv-now").src=s?s.preview:LOG.start.preview;$("pv-now-cap").textContent=k?`after image ${k}`:"start";
  $("stats").innerHTML=s?[["Loss while learning",fmt(s.loss)],["Loss before learning it",fmt(s.loss_before)],["Pixel accuracy before",fmt(100*s.acc_before,1)+"%"],["Cauliflower IoU before",s.iou_cauli_before==null?"–":fmt(s.iou_cauli_before,2)],["Learning rate",s.lr.toExponential(2)],["Largest change",(()=>{const b=B.reduce((a,x)=>s.blocks[x].step>s.blocks[a].step?x:a,B[0]);return b+" "+(100*s.blocks[b].step).toPrecision(2)+"%"})()]].map(([a,b])=>`<dt>${a}</dt><dd>${b}</dd>`).join(""):"<dt>Untrained network</dt><dd></dd>";
  charts.forEach(c=>c.cursor());drawHeat();
}
$("k").max=N;
$("k").addEventListener("input",e=>{k=+e.target.value;update()});
let playing=null;
$("play").onclick=()=>{if(playing){clearInterval(playing);playing=null;$("play").textContent="Play";return}
  if(k>=N)k=0;$("play").textContent="Pause";const dt=matchMedia("(prefers-reduced-motion: reduce)").matches?600:180;
  playing=setInterval(()=>{if(k>=N){clearInterval(playing);playing=null;$("play").textContent="Play";return}k++;update()},dt)};

// line charts
const tip=$("tip");
function pctTxt(v,n){const p=100*v;return p>=10**(n-1)?fmt(p,n>1?2:1):String(+p.toPrecision(n))}
function lineChart(id,{title,note,series,log=false,pct=false}){
  const el=$(id);el.innerHTML=`<h2>${title}</h2><p class="note">${note}</p>`;
  const W=520,H=230,L=46,R=10,T=8,Bm=26,ns="http://www.w3.org/2000/svg";
  const svg=document.createElementNS(ns,"svg");svg.setAttribute("viewBox",`0 0 ${W} ${H}`);el.append(svg);
  const vals=series.flatMap(s=>s.values).filter(v=>v!=null&&(!log||v>0));
  let lo=Math.min(...vals),hi=Math.max(...vals);if(!log){lo=Math.min(0,lo);hi=hi*1.05||1}
  const ly=v=>log?Math.log10(v):v;const a=ly(log?lo*0.8:lo),b=ly(log?hi*1.25:hi);
  const X=i=>L+(W-L-R)*(i-1)/Math.max(N-1,1),Y=v=>T+(H-T-Bm)*(1-(ly(v)-a)/(b-a||1));
  let g="";
  const ticks=log?(()=>{const t=[];for(let e=Math.floor(a);e<=Math.ceil(b);e++)if(e>=a-1e-9&&e<=b+1e-9)t.push(10**e);return t.length>=2?t:[lo,hi]})():[0,1,2,3,4].map(i=>lo+(hi-lo)*i/4);
  for(const t of ticks){const y=Y(t);g+=`<line x1="${L}" x2="${W-R}" y1="${y}" y2="${y}" stroke="var(--grid)" stroke-width="1"/><text x="${L-6}" y="${y+4}" text-anchor="end" font-size="11" fill="var(--muted)">${pct?pctTxt(t,1)+"%":(log?t.toExponential(0):fmt(t,2))}</text>`}
  let prev=1;S.forEach((s,i)=>{if(s.epoch!==prev){const x=X(i+1)-((W-L-R)/Math.max(N-1,1))/2;g+=`<line x1="${x}" x2="${x}" y1="${T}" y2="${H-Bm}" stroke="var(--axis)" stroke-dasharray="3 3"/><text x="${x+4}" y="${T+10}" font-size="11" fill="var(--muted)">epoch ${s.epoch}</text>`;prev=s.epoch}});
  g+=`<line x1="${L}" x2="${W-R}" y1="${H-Bm}" y2="${H-Bm}" stroke="var(--axis)"/>`;
  [1,Math.round(N/2),N].forEach(i=>g+=`<text x="${X(i)}" y="${H-8}" text-anchor="middle" font-size="11" fill="var(--muted)">${i}</text>`);
  g+=`<text x="${W-R}" y="${H-8}" text-anchor="end" font-size="11" fill="var(--muted)"></text>`;
  for(const s of series){let d="";s.values.forEach((v,i)=>{if(v==null||(log&&v<=0))return;d+=(d?"L":"M")+X(i+1).toFixed(1)+","+Y(v).toFixed(1)});g+=`<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`}
  g+=`<line id="${id}-cur" y1="${T}" y2="${H-Bm}" stroke="var(--ink2)" stroke-width="1"/><g id="${id}-dots"></g><rect x="${L}" y="${T}" width="${W-L-R}" height="${H-T-Bm}" fill="transparent" id="${id}-hit" style="cursor:pointer"/>`;
  svg.innerHTML=g;
  if(series.length>1){const lg=document.createElement("div");lg.className="legend";lg.innerHTML=series.map(s=>`<span style="--c:${s.color}">${s.name}</span>`).join("");el.append(lg)}
  const hit=svg.querySelector(`#${id}-hit`);
  const idx=e=>{const r=svg.getBoundingClientRect(),x=(e.clientX-r.left)*W/r.width;return Math.max(1,Math.min(N,Math.round(1+(x-L)/((W-L-R)/Math.max(N-1,1)))))};
  hit.addEventListener("pointermove",e=>{const i=idx(e),s=S[i-1];tip.hidden=false;
    tip.innerHTML=`<div><b>Image ${i}</b> · epoch ${s.epoch}</div><div style="color:var(--muted)">${s.image}</div>`+series.map(sr=>`<div class="r" style="--c:${sr.color}"><i></i>${sr.name}<span style="margin-left:auto;padding-left:10px">${pct?pctTxt(sr.values[i-1],3)+"%":fmt(sr.values[i-1],4)}</span></div>`).join("");
    const tw=tip.offsetWidth;tip.style.left=Math.min(innerWidth-tw-8,e.clientX+14)+"px";tip.style.top=(e.clientY+14)+"px";
    const dots=svg.querySelector(`#${id}-dots`);dots.innerHTML=series.map(sr=>{const v=sr.values[i-1];return v==null||(log&&v<=0)?"":`<circle cx="${X(i)}" cy="${Y(v)}" r="4" fill="${sr.color}" stroke="var(--surface)" stroke-width="2"/>`}).join("")});
  hit.addEventListener("pointerleave",()=>{tip.hidden=true;svg.querySelector(`#${id}-dots`).innerHTML=""});
  hit.addEventListener("click",e=>{k=idx(e);update()});
  return{cursor(){const c=svg.querySelector(`#${id}-cur`);const x=k?X(k):L;c.setAttribute("x1",x);c.setAttribute("x2",x);c.style.opacity=k?1:0}};
}
const blockSeries=f=>B.map((b,i)=>({name:b,color:COL[i],values:S.map(s=>f(s.blocks[b]))}));
const charts=[
  lineChart("c-loss",{title:"Loss",note:"While learning from each image, and on that image just before learning from it. The second line tracks how well the network handles an image it has not seen yet in this epoch.",
    series:[{name:"while learning",color:"var(--s1)",values:S.map(s=>s.loss)},{name:"before learning it",color:"var(--s2)",values:S.map(s=>s.loss_before)}]}),
  lineChart("c-drift",{title:"Distance from the starting weights",note:"‖w − w₀‖ / ‖w₀‖ for each block. Flattening lines mean the block has settled.",series:blockSeries(x=>x.drift),pct:true}),
  lineChart("c-step",{title:"Change made by each image",note:"‖wₖ − wₖ₋₁‖ / ‖wₖ₋₁‖, log scale. Follows the learning-rate schedule; spikes mark images that moved the network unusually far.",series:blockSeries(x=>x.step),log:true,pct:true}),
  lineChart("c-grad",{title:"Gradient size",note:"Root-mean-square gradient norm per block over the steps on each image, log scale. Vanishing or exploding gradients show up here first.",series:blockSeries(x=>x.grad),log:true}),
];

// heatmap
B.forEach(b=>{const o=document.createElement("option");o.value=b;o.textContent=b;$("hb").append(o)});
$("hb").onchange=()=>{drawHeat();fillTable()};
function drawHeat(){const b=$("hb").value||B[0],cv=$("heat"),E=LOG.hist_edges[b],nb=E.length-1;cv.width=N;cv.height=nb;
  const ctx=cv.getContext("2d"),im=ctx.createImageData(N,nb);const lo=hex2rgb(css("--surface")),hi=hex2rgb(css("--s1"));
  const mx=Math.max(...S.flatMap(s=>s.blocks[b].hist));
  S.forEach((s,x)=>s.blocks[b].hist.forEach((c,j)=>{const t=Math.log1p(c)/Math.log1p(mx),y=nb-1-j,p=(y*N+x)*4;for(let q=0;q<3;q++)im.data[p+q]=Math.round(lo[q]+(hi[q]-lo[q])*t);im.data[p+3]=255}));
  if(k){for(let j=0;j<nb;j++){const p=(j*N+k-1)*4;im.data[p]=im.data[p]*.6;im.data[p+1]=im.data[p+1]*.6;im.data[p+2]=im.data[p+2]*.6}}
  ctx.putImageData(im,0,0);$("hy").innerHTML=[E[nb],0,E[0]].map(v=>`<span>${fmt(v,3)}</span>`).join("")}
function fillTable(){const b=$("hb").value||B[0];$("tbl").innerHTML="<tr><th>image</th><th>epoch</th><th>norm</th><th>drift %</th><th>change %</th><th>gradient</th></tr>"+
  S.map(s=>{const x=s.blocks[b];return`<tr><td>${s.k}</td><td>${s.epoch}</td><td>${fmt(x.norm,3)}</td><td>${fmt(100*x.drift,2)}</td><td>${fmt(100*x.step,3)}</td><td>${fmt(x.grad,4)}</td></tr>`}).join("")}
fillTable();update();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change",update);
</script></body></html>"""


def build(log):
    return TEMPLATE.replace("__LOG__", json.dumps(log, separators=(",", ":")))
