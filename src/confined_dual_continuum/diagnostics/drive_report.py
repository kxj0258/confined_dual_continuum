"""协议 5 离线报告：稳定坐标、模型误差和完整轨迹统计，没有合格判据。"""
import argparse
import csv
import html
import json
from pathlib import Path
import numpy as np
from confined_dual_continuum.diagnostics.workspace_visual import workspace_statistics, CONTROLS, SCRIPT


def create_report(directories, destination):
    rows, runs, curves = [], [], []
    velocity_peak_mps = 0.
    for directory in directories:
        directory = Path(directory).resolve()
        summary = json.loads((directory/'summary.json').read_text(encoding='utf-8'))
        if summary.get('protocol_version') != 5:
            raise ValueError('该报告只接受协议 5；旧版结果请使用保存的 Git 版本单独读取')
        runs.append(dict(path=str(directory), **summary))
        for line in (directory/'samples.jsonl').read_text(encoding='utf-8').splitlines():
            row = dict(json.loads(line), run=directory.name)
            rows.append(row)
            if row['kind'] in ('dynamic', 'response'):
                # 图形适度抽样；原始 trace 和记录内全部周期统计不受影响。
                trace_path = directory/f"{row['sample']:04d}_trace.jsonl"
                with trace_path.open(encoding='utf-8') as stream:
                    count = sum(1 for _ in stream)
                stride = max(1, int(np.ceil(count/800)))
                points = []
                with trace_path.open(encoding='utf-8') as stream:
                    for index, record in enumerate(stream):
                        r = json.loads(record)
                        # 共用范围依据全部原始周期、全部路径和肌腱，避免抽样漏掉峰值。
                        for key in ('requested_velocity_mps', 'sent_velocity_mps', 'actual_velocity_mps'):
                            velocity_peak_mps = max(velocity_peak_mps, max(abs(v) for v in r[key]))
                        if index % stride == 0 or index == count-1:
                            points.append({k: r[k] for k in ('velocity_time_s', 'acceleration_time_s', 'leg', 'phase',
                                'requested_velocity_mps', 'sent_velocity_mps', 'actual_velocity_mps',
                                'requested_acceleration_mps2', 'sent_acceleration_mps2', 'actual_acceleration_mps2')})
                curves.append(dict(label=f"{directory.name}/{row['sample']} {row['arm']} {row['speed_mps']} m/s",
                                   sample=row, points=points, trace=str(trace_path)))
    workspace = []
    for run in runs:
        workspace.extend(dict(g, run=Path(run['path']).name) for g in run['workspace'])
    dynamic = [dict(run=r['run'], sample=r['sample'], arm=r['arm'], kind=r['kind'],
                    speed_mps=r['speed_mps'], acceleration_mps2=r['acceleration_mps2'], status=r['status'],
                    peak_actual_speed_mps=r['peak_actual_speed_mps'],
                    peak_actual_acceleration_mps2=r['peak_actual_acceleration_mps2'])
               for r in rows if r['kind'] in ('dynamic', 'response')]
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    velocity_axis_limit_mm_s = max(velocity_peak_mps * 1000 * 1.05, .001)
    workspace = workspace_statistics([r for r in rows if r['kind'] == 'workspace'])
    result = dict(runs=runs, workspace=workspace, dynamic=dynamic,
                  velocity_axis_mm_s=[-velocity_axis_limit_mm_s, velocity_axis_limit_mm_s])
    (destination/'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    with (destination/'samples.jsonl').open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    events = []
    for r in rows:
        for leg in r['legs']:
            events.append(dict(run=r['run'], sample=r['sample'], arm=r['arm'], kind=r['kind'],
                               leg=leg['leg'], status=leg['status'], sent_s=leg['sent_s'], stable_s=leg['stable_s'],
                               duration_s=leg['duration_s'], planned_motion_s=leg['planned_motion_s'],
                               planned_peak_speed_mps=leg['planned_peak_speed_mps'], planned_cruise_s=leg['planned_cruise_s']))
    if events:
        with (destination/'events.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(events[0]))
            writer.writeheader()
            writer.writerows(events)
    def table(items):
        if not items:
            return '<p>本次未采集。</p>'
        keys = list(items[0])
        return '<table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in keys)+'</tr>'+''.join(
            '<tr>'+''.join('<td>'+html.escape(str(r[k]))+'</td>' for k in keys)+'</tr>' for r in items)+'</table>'
    space = [r for r in rows if r['kind'] == 'workspace']
    page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>工作空间与驱动轨迹记录</title>
<style>body{font:15px system-ui;max-width:1300px;margin:30px auto;padding:0 20px;color:#21343d}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:6px}svg{display:block;border:1px solid #ddd;background:#fafafa;max-width:100%;margin:12px 0}select{padding:6px;margin:8px}pre{white-space:pre-wrap;background:#f5f5f5;padding:15px;max-height:500px;overflow:auto}</style>
<h1>工作空间与驱动轨迹记录</h1><p>无长度或速度误差合格阈值。stable 只表示实际静止及保持完成。真实量均来自仿真，不代表硬件测量。</p>
<h2>稳定工作空间</h2>''' + table(workspace) + '''
<select id="arm"><option>executor</option><option>observer</option></select><select id="projection"><option>XY</option><option>XZ</option><option>YZ</option></select><select id="population"><option value="stable">仅稳定点</option><option value="all">全部终态（含中断）</option></select>
<p>坐标为臂根坐标，单位 mm。绿色是稳定点；灰色是中断终态，不代表目标可达。点击点查看完整世界/臂根位姿、指令 PCC 与反馈反算 PCC 相对 MuJoCo 的位置、姿态和中心线误差。</p>
<svg id="space" width="1000" height="400" viewBox="0 0 1000 400"></svg><pre id="spaceDetail"></pre>
<h2>速度与加速度轨迹</h2>''' + table(dynamic) + '''
<select id="sample"></select><select id="tendon"></select>
<p>蓝：请求；橙：实际发送；绿：MuJoCo。横轴为仿真时间 s；速度单位 mm/s，加速度单位 mm/s²。所有速度图使用相同的对称纵轴范围。速度按同一控制区间平均，加速度按相邻区间速度差分；图形最多约800点/路径，原始 trace 完整保留。切换周期可能跨越两个解析阶段。</p>
<h3>速度</h3><svg id="velocity" width="1000" height="300" viewBox="0 0 1000 300"></svg>
<h3>加速度</h3><svg id="acceleration" width="1000" height="300" viewBox="0 0 1000 300"></svg>
<pre id="driveDetail"></pre><script>'''
    page += 'const spaces='+json.dumps(space, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')+';'
    page += 'const curves='+json.dumps(curves, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')+';'
    page += 'const velocityLimit='+json.dumps(velocity_axis_limit_mm_s)+';'
    page += '''
const el=id=>document.getElementById(id), ns='http://www.w3.org/2000/svg';
function node(tag,attrs){let n=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));return n;}
function spaceDraw(){let rs=spaces.filter(r=>r.arm===el('arm').value&&(el('population').value==='all'||r.status==='stable'));let s=el('space');s.replaceChildren();if(!rs.length){el('spaceDetail').textContent='没有对应点';return;}let a='XYZ'.indexOf(el('projection').value[0]),b='XYZ'.indexOf(el('projection').value[1]);let xs=rs.map(r=>r.T_root_tip[a][3]*1000),ys=rs.map(r=>r.T_root_tip[b][3]*1000);let lx=Math.min(...xs),hx=Math.max(...xs),ly=Math.min(...ys),hy=Math.max(...ys);el('spaceDetail').textContent=rs.length+' 点，'+el('projection').value+': '+JSON.stringify([[lx,hx],[ly,hy]])+' mm';rs.forEach((r,i)=>{let c=node('circle',{cx:40+920*(xs[i]-lx)/Math.max(hx-lx,1),cy:360-320*(ys[i]-ly)/Math.max(hy-ly,1),r:4,fill:r.status==='stable'?'#16815e':'#999'});c.onclick=()=>el('spaceDetail').textContent=JSON.stringify(r,null,2);s.appendChild(c);});}
curves.forEach((c,i)=>{let o=document.createElement('option');o.value=i;o.textContent=c.label;el('sample').appendChild(o);});
for(let i=0;i<9;i++){let o=document.createElement('option');o.value=i;o.textContent='肌腱 '+(i+1);el('tendon').appendChild(o);}
function plot(id,pts,keys,timeKey){let s=el(id);s.replaceChildren();if(!pts.length)return;let i=+el('tendon').value;let xs=pts.map(p=>p[timeKey]),ys=keys.flatMap(k=>pts.map(p=>p[k][i]*1000));let lx=Math.min(...xs),hx=Math.max(...xs),ly=id==='velocity'?-velocityLimit:Math.min(0,...ys),hy=id==='velocity'?velocityLimit:Math.max(0,...ys);keys.forEach((k,j)=>{let points=pts.map((p,n)=>(50+920*(xs[n]-lx)/Math.max(hx-lx,.001))+','+(260-220*(p[k][i]*1000-ly)/Math.max(hy-ly,.001))).join(' ');s.appendChild(node('polyline',{points,fill:'none',stroke:['#2474bb','#d97706','#16815e'][j],'stroke-width':1.5}));});let t=node('text',{x:12,y:18,'font-size':12});t.textContent='t: '+lx.toFixed(2)+' — '+hx.toFixed(2)+' s; y: '+ly.toFixed(3)+' — '+hy.toFixed(3);s.appendChild(t);}
function driveDraw(){if(!curves.length){el('driveDetail').textContent='未采集动态路径';return;}let c=curves[+el('sample').value];plot('velocity',c.points,['requested_velocity_mps','sent_velocity_mps','actual_velocity_mps'],'velocity_time_s');plot('acceleration',c.points,['requested_acceleration_mps2','sent_acceleration_mps2','actual_acceleration_mps2'],'acceleration_time_s');el('driveDetail').textContent=JSON.stringify({trace:c.trace,legs:c.sample.legs,tracking:c.sample.tracking},null,2);}
['arm','projection','population'].forEach(id=>el(id).onchange=spaceDraw);['sample','tendon'].forEach(id=>el(id).onchange=driveDraw);spaceDraw();driveDraw();</script></html>'''
    path = destination/'report.html'
    page = page.replace('<option>XY</option>', '<option>3D</option><option>任意二维</option><option>XY</option>')
    page = page.replace('<svg id="space"', CONTROLS+'<svg id="space"')
    start = page.index('function spaceDraw()')
    end = page.index('curves.forEach', start)
    page = page[:start] + SCRIPT + page[end:]
    path.write_text(page, encoding='utf-8')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', nargs='+')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(create_report(args.directories, args.output))


if __name__ == '__main__':
    main()
