#!/usr/bin/env python3
"""Render a portable, offline *hypothetical* accelerator placement explorer.

Input: one or more compiler_lab_demo.py analysis.json files with v3 sweep data.
No hardware measurement and no execution in browser. Plans are precomputed
from the exact binary-cut solver, not recomputed from arbitrary JS data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>VisuaML · Hardware–Algorithm Co-Design</title>
<style>
:root{--base:#090f1b;--panel:#121e30;--line:#2b3d55;--text:#eaf1ff;--muted:#94a9c4;--teal:#61e5bc;--orange:#ffa86e;--blue:#8fb5ff}
*{box-sizing:border-box}body{margin:0;background:var(--base);color:var(--text);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
header{padding:20px 28px;border-bottom:1px solid var(--line);background:linear-gradient(120deg,#111d30,#0d1422)}
h1{margin:0;font-size:23px;letter-spacing:-.7px}h1 span{color:var(--teal)}.sub{color:var(--muted);font-size:12px;margin-top:3px}
.banner{padding:10px 28px;background:#2d261c;color:#ffd9a6;border-bottom:1px solid #57402c;font-size:12px}
.layout{display:grid;grid-template-columns:minmax(0,1fr) 330px;min-height:calc(100vh - 140px)} main{min-width:0;padding:22px 25px;border-right:1px solid var(--line)} aside{padding:22px;background:#101a29}
label{display:block;color:var(--muted);font-weight:650;text-transform:uppercase;font-size:11px;letter-spacing:.9px;margin-bottom:6px}
.controls{display:flex;gap:18px;flex-wrap:wrap;align-items:end}select{min-width:180px;border-radius:8px;border:1px solid #3a5374;background:#15253c;color:var(--text);padding:9px 11px;font-size:14px}
.penalties{display:flex;gap:6px;flex-wrap:wrap}.penalty{border:1px solid var(--line);background:#142337;color:#d3e2f7;border-radius:7px;padding:9px 12px;cursor:pointer}
.penalty.active{border-color:var(--teal);background:#123c38;color:#e2fff3}
.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:22px 0}.metric{padding:15px;background:var(--panel);border:1px solid var(--line);border-radius:11px}
.metric .name{font-size:11px;color:var(--muted)}.metric .value{font-size:21px;font-weight:750;margin-top:2px}
.sheading{font-size:12px;font-weight:800;text-transform:uppercase;letter-spacing:.9px;color:var(--muted);margin:24px 0 11px}
.graph{overflow:auto;background-color:#0b1829;background-image:linear-gradient(#253246 1px,transparent 1px),linear-gradient(90deg,#253246 1px,transparent 1px);background-size:26px 26px;border:1px solid var(--line);border-radius:12px;height:500px}
.graph svg{display:block}.legend{margin-top:10px;display:flex;gap:16px;font-size:12px;color:var(--muted)}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:4px}.dev{background:var(--teal)}.host{background:#7797c3}
.barline{margin:13px 0}.barn{display:flex;justify-content:space-between;margin-bottom:4px;color:#b9c9db;font-size:12px}.bar{height:12px;border-radius:4px;background:#27384f;overflow:hidden}.fill{height:100%;background:var(--teal);min-width:1px}.fill.greedy{background:var(--orange)}.fill.base{background:#6686b7}
table{width:100%;border-collapse:collapse;font-size:12px}th{text-align:left;color:#9fb4ce;font-size:11px}td,th{padding:10px 4px;border-bottom:1px solid #2b3c53}td:last-child{text-align:right}
.tag{padding:3px 7px;border-radius:5px;background:#283e53;color:#d1e1f6;font-size:11px}.tag.device{background:#153f36;color:#7dffcb}
.small{font-size:12px;color:var(--muted)}.status{color:var(--teal);font-weight:750}b{color:#e9f3fe}
@media(max-width:980px){.layout{grid-template-columns:1fr}main{border-right:none}aside{border-top:1px solid var(--line)}.metrics{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style></head><body>
<header><h1>Visua<span>ML</span> · Hardware–Algorithm Co-Design</h1><div class="sub">Graph decomposition / exact binary-cost partitioning / executable cross-backend witness checks</div></header>
<div class="banner"><b>Scientific boundary:</b> This uses explicitly fictional accelerator costs, not Ayo processor specifications. “Savings” are abstract model units, not measured speedups, watts, or physical execution. NumPy-AOT is the executed backend; these partitions are proposed placements only.</div>
<div class="layout"><main>
<div class="controls"><div><label for="scenario">Computation workload</label><select id="scenario"></select></div><div><label>Boundary transfer overhead (abstract)</label><div class="penalties" id="penalties"></div></div></div>
<div class="metrics"><div class="metric"><div class="name">Operators on hypothetical device</div><div class="value" id="placed">—</div></div><div class="metric"><div class="name">Cut boundary edges</div><div class="value" id="edges">—</div></div><div class="metric"><div class="name">Model-estimated saving</div><div class="value" id="saving">—</div></div><div class="metric"><div class="name">Cross-backend numerical check</div><div class="value status" id="parity">—</div></div></div>
<div class="sheading">Computational graph · host/device placement</div><div class="graph"><svg id="graph"></svg></div>
<div class="legend"><span><i class="dot dev"></i>Hypothetical accelerator</span><span><i class="dot host"></i>Host / fallback</span><span>Click nodes for inspection</span></div>
</main><aside><div class="sheading" style="margin-top:0">Cost objective (lower is better)</div><div id="bars"></div><p class="small" id="explain"></p>
<div class="sheading">Selected node</div><div id="selected" class="small">Click a node in the graph to inspect operator, tensor, and modeled execution costs.</div>
<div class="sheading">Boundary transfers</div><table><thead><tr><th>Producer → Consumer</th><th>Bytes</th><th>Units</th></tr></thead><tbody id="transfers"></tbody></table>
<div class="sheading">Provenance / limitations</div><p class="small">Each scenario contains trace stages, numeric pass witnesses, a portable NumPy artifact, a PyTorch export, and a graph fingerprint. This view reads recorded solver results; it does not invoke external services or execute model code.</p></aside></div>
<script type="application/json" id="data">__DATA__</script><script>
(()=>{'use strict';
const data=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id);
let scenario=Object.keys(data)[0],pi=0;
const sel=$('scenario');for(let name of Object.keys(data)){let o=document.createElement('option');o.value=name;o.textContent=name.toUpperCase();sel.appendChild(o)};
sel.addEventListener('change',()=>{scenario=sel.value;pi=0;render()});
const fmt=(n,d=2)=>Number(n).toLocaleString(undefined,{maximumFractionDigits:d});
function render(){let report=data[scenario],sw=report.hypothetical_partition_sweep;pi=Math.min(pi,sw.length-1);let p=sw[pi].partition;
$('penalties').replaceChildren(...sw.map((v,i)=>{let b=document.createElement('button');b.className='penalty'+(i===pi?' active':'');b.textContent=fmt(v.transfer_overhead);b.addEventListener('click',()=>{pi=i;render()});return b}));
let mapping=Object.fromEntries(p.assignments.map(v=>[v.node,v]));$('placed').textContent=p.assignments.filter(v=>v.assigned==='device').length;$('edges').textContent=p.boundaries.length;
let sav=p.savings_vs_all_host/p.all_host_baseline*100;$('saving').textContent=fmt(sav,1)+'%';
$('parity').textContent=report.witness.passed?'PASS':'FAIL';$('explain').textContent=`Exact optimal cut for the declared additive model. Transfer overhead ${fmt(sw[pi].transfer_overhead)} per crossing + bytes / 512; each supported accelerator op receives a synthetic cost of 2 units. Not an actual physical benchmark.`;
$('bars').replaceChildren();let mx=Math.max(p.all_host_baseline,p.unconstrained_greedy_cost,p.estimated_total_cost,1);
for(let [n,v,klass] of [['All-host baseline',p.all_host_baseline,'base'],['Per-node greedy',p.unconstrained_greedy_cost,'greedy'],['Exact min-cut',p.estimated_total_cost,'']]){
let line=document.createElement('div');line.className='barline';let title=document.createElement('div');title.className='barn';let a=document.createElement('span');a.textContent=n;let b=document.createElement('b');b.textContent=fmt(v)+' units';title.append(a,b);let track=document.createElement('div');track.className='bar';let fill=document.createElement('div');fill.className='fill '+klass;fill.style.width=(100*v/mx)+'%';track.appendChild(fill);line.append(title,track);$('bars').append(line)}
$('transfers').replaceChildren();for(let edge of p.boundaries){let tr=document.createElement('tr');for(let value of [edge.producer+' → '+edge.consumer,fmt(edge.bytes,0),fmt(edge.estimated_cost)]){let td=document.createElement('td');td.textContent=value;tr.append(td)}$('transfers').append(tr)}
let graph=report.stages.at(-1).canvas, nodes=graph.nodes, edges=graph.edges, byId=Object.fromEntries(nodes.map(n=>[n.id,n]));let depth={};for(let n of nodes){let parents=edges.filter(e=>e.target===n.id);depth[n.id]=parents.length?Math.max(...parents.map(e=>(depth[e.source]??0)+1)):0}
let layers={};for(let n of nodes)(layers[depth[n.id]]??=[]).push(n);let mw=Math.max(...Object.values(layers).map(x=>x.length)),W=Math.max(900,(Object.keys(layers).length+1)*194),H=Math.max(480,mw*91+70),pos={};for(let [column,items] of Object.entries(layers))for(let i=0;i<items.length;i++)pos[items[i].id]={x:30+Number(column)*194,y:38+i*91+(mw-items.length)*46};
let svg=$('graph');svg.replaceChildren();svg.setAttribute('width',W);svg.setAttribute('height',H);
function el(type,attrs,text){let x=document.createElementNS('http://www.w3.org/2000/svg',type);for(let [k,v] of Object.entries(attrs))x.setAttribute(k,v);if(text)x.textContent=text;return x}
for(let e of edges){let a=pos[e.source],b=pos[e.target];if(!a||!b)continue;let cross=mapping[e.source]?.assigned!==mapping[e.target]?.assigned;
svg.append(el('path',{d:`M${a.x+154},${a.y+30} C${a.x+175},${a.y+30} ${b.x-17},${b.y+30} ${b.x},${b.y+30}`,stroke:cross?'#f1b573':'#537491','stroke-width':cross?2.7:1.3,fill:'none'}))}
for(let n of nodes){let p0=pos[n.id], info=mapping[n.id],isDev=info?.assigned==='device';let group=el('g',{transform:`translate(${p0.x},${p0.y})`,style:'cursor:pointer'});group.append(el('rect',{width:154,height:62,rx:8,fill:isDev?'#17463a':'#1a2b44',stroke:isDev?'#61e5bc':'#6687b0','stroke-width':1.3}));group.append(el('text',{x:9,y:24,fill:'#f3f7ff','font-size':12,'font-weight':'bold'},(n.data.label||n.id).slice(0,19)));group.append(el('text',{x:9,y:43,fill:isDev?'#7affcb':'#a4c1e2','font-size':11},n.data.op+' · '+(isDev?'device':'host')));group.addEventListener('click',()=>{let box=$('selected');box.replaceChildren();for(let [k,v] of [['Node',n.id],['Operator',n.data.op],['Output',n.data.outputShape||'unknown'],['Placement',info?.assigned||'unknown'],['Host model',info?.host_cost],['Device model',info?.device_cost??'not available']]){let row=document.createElement('p');let strong=document.createElement('b');strong.textContent=k+': ';row.append(strong,document.createTextNode(String(v)));box.append(row)}});svg.append(group)}
}render();})();
</script></body></html>"""


def build(reports: list[Path], output: Path) -> Path:
    data = {}
    for path in reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        if not report.get("hypothetical_partition_sweep"):
            raise ValueError(f"Report lacks v3 placement profiles: {path}")
        data[report["scenario"]] = report
    if not data:
        raise ValueError("At least one report is required")
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False).replace(
        "<", "\\u003c"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(HTML.replace("__DATA__", payload), encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.analysis, args.out))
