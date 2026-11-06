#!/usr/bin/env python3
"""Build an offline VisuaML-like interactive inspector from analysis.json.

No external JS/CDN and no model execution in the browser. This is a useful
presentation surface and integration design reference, not a modification
of VisuaML's existing React frontend.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>VisuaML / Compiler Laboratory</title>
<style>
:root{--bg:#0a1019;--panel:#111b28;--border:#2b3c4f;--text:#e8f0f8;--muted:#90a5b9;--teal:#50e3c2;--amber:#e5b66f;--blue:#7eacf3;--red:#f18a8a}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 ui-sans-serif,system-ui,-apple-system,'Segoe UI',sans-serif}
.top{display:flex;align-items:center;justify-content:space-between;gap:20px;flex-wrap:wrap;padding:22px 25px 14px;border-bottom:1px solid var(--border);background:#0d1622}
.logo{font-size:20px;font-weight:780;letter-spacing:-.6px}.logo span{color:var(--teal)} .sub{color:var(--muted);font-size:12px;margin-top:1px;letter-spacing:.6px;text-transform:uppercase}
.pills{display:flex;gap:7px;flex-wrap:wrap}.pill{padding:6px 10px;border:1px solid var(--border);border-radius:20px;font-size:12px;color:#c4d5e7}
.pill.good{color:var(--teal);border-color:#3d796e}.main{display:grid;grid-template-columns:minmax(0,1fr) 325px;min-height:calc(100vh - 85px)}
.left{min-width:0;border-right:1px solid var(--border)}.stagebox{padding:16px 18px 10px}.eyebrow{color:var(--muted);font-size:11px;letter-spacing:1px;font-weight:800;text-transform:uppercase}
.stages{display:flex;gap:8px;margin:9px 0;flex-wrap:wrap}.stage{border:1px solid var(--border);color:#b7c8da;background:#111c2b;padding:8px 12px;border-radius:7px;cursor:pointer;font:inherit}
.stage.active{background:#123b3b;border-color:#4ab5a7;color:#d7fff4}.stage:hover{border-color:var(--teal)}
.stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin:13px 0 4px}.stat{background:#111b28;border:1px solid #273849;border-radius:8px;padding:11px 13px}
.stat .label{color:var(--muted);font-size:11px}.stat .value{font-size:18px;font-weight:750;margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.graph-header{display:flex;justify-content:space-between;align-items:center;padding:11px 18px;border-top:1px solid var(--border);border-bottom:1px solid var(--border);gap:12px}
.graph-header .hint{color:var(--muted);font-size:12px}.graph-scroll{height:min(62vh,690px);min-height:410px;overflow:auto;background-color:#0a1320;background-image:linear-gradient(#223244 1px,transparent 1px),linear-gradient(90deg,#223244 1px,transparent 1px);background-size:22px 22px}
#canvas{display:block}.details{padding:20px;position:sticky;top:0;align-self:start;max-height:100vh;overflow:auto}.details h2{margin:5px 0 12px;font-size:20px;word-break:break-word}
.kv{display:grid;grid-template-columns:100px 1fr;gap:11px 8px;padding:13px 0;border-bottom:1px solid #233449;font-size:12px}.kv dt{color:var(--muted)}.kv dd{margin:0;overflow-wrap:anywhere}
.pcap{margin-top:22px}.receipt{background:#132132;border:1px solid var(--border);border-radius:9px;padding:11px;margin-top:8px}
.receipt .name{font-weight:700;color:var(--teal)}.receipt .desc{font-size:11px;color:#a6b6c7;margin-top:3px}.foot{color:#9aadc0;padding:14px 18px 22px;border-top:1px solid var(--border);font-size:12px}
svg text{font-family:ui-sans-serif,system-ui;pointer-events:none}.node{cursor:pointer}.node:hover rect{stroke:#fff;stroke-width:2}.edge{stroke:#49657a;stroke-width:1.6;fill:none;opacity:.7}.node.selected rect{stroke:var(--teal);stroke-width:2.3}
@media(max-width:900px){.main{grid-template-columns:1fr}.left{border-right:none}.details{position:static;border-top:1px solid var(--border);max-height:none}.stats{grid-template-columns:repeat(2,1fr)}}
</style></head>
<body><header class="top"><div><div class="logo">Visua<span>ML</span> / compiler laboratory</div><div class="sub">Executable decomposition • verified passes • backend lowering</div></div><div class="pills"><span class="pill good" id="status">Witness parity</span><span class="pill">Torch .pt2</span><span class="pill">NumPy AOT</span><span class="pill" id="scenario"></span></div></header>
<div class="main"><section class="left"><div class="stagebox"><div class="eyebrow">Transformation history</div><div id="stageControls" class="stages"></div><div class="stats">
<div class="stat"><div class="label">Captured nodes</div><div class="value" id="nodeCount">—</div></div><div class="stat"><div class="label">Known FLOPs (partial)</div><div class="value" id="flops">—</div></div><div class="stat"><div class="label">Peak live bytes (est.)</div><div class="value" id="memory">—</div></div><div class="stat"><div class="label">Graph ID</div><div class="value" id="digest">—</div></div></div></div>
<div class="graph-header"><div class="eyebrow">Typed computational graph</div><span class="hint">Scroll horizontally • select a node to inspect</span></div><div class="graph-scroll"><svg id="canvas" role="img" aria-label="Computational graph"></svg></div>
<div class="foot">Illustrative static cost estimates. Floating-point witnesses only establish agreement for tested inputs. This is not a validated optical device simulator.</div></section>
<aside class="details"><div class="eyebrow">Node inspector</div><h2 id="nodeTitle">Select an operation</h2><dl class="kv" id="nodeInfo"><dt>Tip</dt><dd>Click any graph node for tensor and placement details.</dd></dl>
<div class="pcap"><div class="eyebrow">Pass evidence</div><div id="receipts"></div></div><div class="pcap"><div class="eyebrow">Backend verification</div><div id="checks"></div></div>
</aside></div>
<script type="application/json" id="payload">__PAYLOAD__</script>
<script>
(() => {'use strict';
const report = JSON.parse(document.getElementById('payload').textContent);let stageIndex=0,selected=null;
const $=id=>document.getElementById(id);$('scenario').textContent=report.scenario.toUpperCase();
const NODE_W=160,NODE_H=65,GAP_X=210,GAP_Y=108;
function el(tag,attributes={},text=''){let n=document.createElementNS('http://www.w3.org/2000/svg',tag);for(let [k,v] of Object.entries(attributes))n.setAttribute(k,v);if(text)n.textContent=text;return n;}
function fmt(n){return Number(n||0).toLocaleString()};
function inspect(n){selected=n.id;$('nodeTitle').textContent=n.data.label+' · '+n.id;let rows=[['Operator',n.data.op],['Target',n.data.target],['Output shape',n.data.outputShape||'unknown'],['Est. FLOPs',n.data.estimatedFlops===null?'unknown':fmt(n.data.estimatedFlops)],['Placement',n.data.backend||'not planned'],['Reason',n.data.loweringReason||'—'],['Decomposition',(n.data.decomposition||[]).join(' → ')||'—']];$('nodeInfo').replaceChildren(...rows.flatMap(([k,v])=>{let a=document.createElement('dt'),b=document.createElement('dd');a.textContent=k;b.textContent=v;return[a,b]}));}
function buildGraph(stage){let nodes=stage.canvas.nodes,edges=stage.canvas.edges;let depth={};for(let n of nodes){let ins=edges.filter(e=>e.target===n.id);depth[n.id]=ins.length?1+Math.max(...ins.map(e=>depth[e.source]||0)):0;}let layers={};nodes.forEach(n=>(layers[depth[n.id]]??=[]).push(n));let maxDepth=Math.max(...Object.keys(layers).map(Number)),maxLayer=Math.max(...Object.values(layers).map(x=>x.length));let width=Math.max(920,(maxDepth+1)*GAP_X+80),height=Math.max(420,maxLayer*GAP_Y+65),pos={};for(let [lev,items] of Object.entries(layers)){items.forEach((n,i)=>{pos[n.id]={x:36+Number(lev)*GAP_X,y:40+i*GAP_Y+(maxLayer-items.length)*GAP_Y/2};});}
let svg=$('canvas');svg.replaceChildren();svg.setAttribute('width',width);svg.setAttribute('height',height);for(let e of edges){let a=pos[e.source],b=pos[e.target];if(!a||!b)continue;svg.appendChild(el('path',{class:'edge',d:`M ${a.x+NODE_W} ${a.y+NODE_H/2} C ${a.x+NODE_W+42} ${a.y+NODE_H/2}, ${b.x-42} ${b.y+NODE_H/2}, ${b.x} ${b.y+NODE_H/2}`}));}
for(let n of nodes){let p=pos[n.id];let g=el('g',{class:'node'+(selected===n.id?' selected':''),transform:`translate(${p.x},${p.y})`});let c=n.data.backend==='numpy-aot'?'#225a58':n.data.op==='input'?'#253c61':n.data.op==='output'?'#58452c':'#1b3446';g.appendChild(el('rect',{width:NODE_W,height:NODE_H,rx:9,fill:c,stroke:'#648397','stroke-width':1}));g.appendChild(el('text',{x:12,y:25,fill:'#f2f8ff','font-size':13,'font-weight':700},n.data.label.slice(0,20)));g.appendChild(el('text',{x:12,y:46,fill:'#b3cada','font-size':11},(n.data.outputShape||n.id).slice(0,24)));g.addEventListener('click',()=>{inspect(n);document.querySelectorAll('.node').forEach(t=>t.classList.remove('selected'));g.classList.add('selected')});svg.appendChild(g);}
}
function render(){let st=report.stages[stageIndex],c=st.cost;$('nodeCount').textContent=fmt(c.node_count);$('flops').textContent=fmt(c.known_flops);$('memory').textContent=fmt(c.peak_live_activation_bytes);$('digest').textContent=st.graph_sha256.slice(0,10)+'…';$('stageControls').replaceChildren(...report.stages.map((s,i)=>{let b=document.createElement('button');b.className='stage'+(i===stageIndex?' active':'');b.textContent=(i+1)+'. '+s.name.replaceAll('_',' ');b.addEventListener('click',()=>{stageIndex=i;selected=null;render()});return b}));$('nodeTitle').textContent='Select an operation';let label=document.createElement('dt'),value=document.createElement('dd');label.textContent='Stage';value.textContent=st.name.replaceAll('_',' ');$('nodeInfo').replaceChildren(label,value);$('checks').replaceChildren();for(let [name,d] of [['NumPy AOT',report.witness],['Torch export',report.torch_export_roundtrip]]){let card=document.createElement('div');card.className='receipt';let err=Math.max(...d.leaves.map(x=>x.max_abs_error));card.textContent=`${name} · ${d.passed?'PASS':'FAIL'} · max abs ${err.toExponential(2)}`;$('checks').appendChild(card)}$('receipts').replaceChildren();for(let r of report.passes){let card=document.createElement('div');card.className='receipt';let a=document.createElement('div');a.className='name';a.textContent=r.pass_name+' · '+(r.passed?'PASS':'FAIL');let b=document.createElement('div');b.className='desc';b.textContent=`${r.before_nodes}→${r.after_nodes} nodes · ${r.witnesses_checked} witnesses · max abs ${Number(r.max_abs_error).toExponential(2)}`;card.append(a,b);$('receipts').appendChild(card)}buildGraph(st);}
render();})();
</script></body></html>"""


def generate(analysis: Path, target: Path) -> Path:
    data = json.loads(analysis.read_text(encoding="utf-8"))
    if data.get("schema") != "visuaml.compiler-lab.report.v2":
        raise ValueError("Unsupported compiler report")
    payload = json.dumps(data, sort_keys=True, allow_nan=False).replace("<", "\\u003c")
    target.write_text(HTML.replace("__PAYLOAD__", payload), encoding="utf-8")
    return target


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("analysis", type=Path)
    p.add_argument("target", type=Path)
    a = p.parse_args()
    print(generate(a.analysis, a.target))
