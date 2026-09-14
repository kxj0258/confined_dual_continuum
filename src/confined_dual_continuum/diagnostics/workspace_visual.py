"""离线工作空间统计与等比例三维/任意平面投影，不依赖网络绘图库。"""
import numpy as np


def workspace_statistics(rows):
    result = []
    for run, scope, arm in sorted({(r['run'], r['scope'], r['arm']) for r in rows}):
        group = [r for r in rows if (r['run'], r['scope'], r['arm']) == (run, scope, arm)]
        stable = [r for r in group if r['status'] == 'stable']
        item = dict(run=run, scope=scope, arm=arm, sampled=len(group), stable=len(stable),
                    stable_fraction=len(stable)/len(group),
                    status_counts={s: sum(r['status'] == s for r in group) for s in sorted({r['status'] for r in group})})
        if stable:
            xyz = np.array([np.asarray(r['T_root_tip'])[:3, 3]*1000 for r in stable])
            item.update(min_xyz_mm=xyz.min(axis=0).tolist(), max_xyz_mm=xyz.max(axis=0).tolist(),
                        span_xyz_mm=np.ptp(xyz, axis=0).tolist(), mean_xyz_mm=xyz.mean(axis=0).tolist())
            for source in ('command', 'estimated'):
                for metric in ('tip_error_m', 'orientation_error_deg', 'centerline_rms_m', 'centerline_max_m'):
                    values = np.array([r['comparison'][source][metric] for r in stable])
                    item[source+'_'+metric] = dict(mean=float(values.mean()), p95=float(np.percentile(values, 95)), maximum=float(values.max()))
        result.append(item)
    return result


CONTROLS = '''
<label>方位角 <input id="azimuth" type="range" min="-180" max="180" value="35"></label>
<label>仰角 <input id="elevation" type="range" min="-90" max="90" value="25"></label>
<button id="saveSpace">导出当前 SVG</button>
<p>3D模式可拖动旋转；任意二维投影由方位角、仰角定义投影平面的法向。所有方向使用相同毫米比例尺。
统计只针对稳定实测点，包围盒跨度不等于工作空间体积；误差字段 _m 为米，_deg 为度。</p>
'''

SCRIPT = '''
function spaceDraw(){
 const rs=spaces.filter(r=>r.arm===el('arm').value&&(el('population').value==='all'||r.status==='stable'));
 const s=el('space');s.replaceChildren();if(!rs.length){el('spaceDetail').textContent='没有对应点';return;}
 const mode=el('projection').value,az=+el('azimuth').value*Math.PI/180,ev=+el('elevation').value*Math.PI/180;
 let u=[-Math.sin(az),Math.cos(az),0],v=[-Math.sin(ev)*Math.cos(az),-Math.sin(ev)*Math.sin(az),Math.cos(ev)];
 if(mode==='XY'){u=[1,0,0];v=[0,1,0];}if(mode==='XZ'){u=[1,0,0];v=[0,0,1];}if(mode==='YZ'){u=[0,1,0];v=[0,0,1];}
 const xyz=rs.map(r=>r.T_root_tip.slice(0,3).map(row=>row[3]*1000));
 const bounds=[0,1,2].map(j=>[Math.min(...xyz.map(p=>p[j])),Math.max(...xyz.map(p=>p[j]))]);
 const center=bounds.map(b=>(b[0]+b[1])/2),diameter=Math.max(1,Math.hypot(...bounds.map(b=>b[1]-b[0]))),scale=300/diameter;
 const dot=(p,a)=>p.reduce((z,x,j)=>z+x*a[j],0),project=p=>{let d=p.map((x,j)=>x-center[j]);return [500+scale*dot(d,u),200-scale*dot(d,v)];};
 function line(a,b,color){const x=project(a),y=project(b);s.appendChild(node('line',{x1:x[0],y1:x[1],x2:y[0],y2:y[1],stroke:color}));}
 const corners=Array.from({length:8},(_,i)=>bounds.map((b,j)=>b[(i>>j)&1]));
 for(let i=0;i<8;i++)for(let j=0;j<3;j++)if(!(i&(1<<j)))line(corners[i],corners[i|(1<<j)],'#ccd3d8');
 const origin=bounds.map(b=>b[0]);for(let j=0;j<3;j++){let p=origin.slice();p[j]=bounds[j][1];line(origin,p,['#c44','#398a55','#367dcc'][j]);let t=node('text',{x:project(p)[0]+5,y:project(p)[1]-5,'font-size':12});t.textContent='XYZ'[j]+' '+bounds[j][1].toFixed(1)+' mm';s.appendChild(t);}
 const normal=[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]];
 rs.map((r,i)=>({r,p:xyz[i]})).sort((a,b)=>dot(a.p,normal)-dot(b.p,normal)).forEach(({r,p})=>{let q=project(p),c=node('circle',{cx:q[0],cy:q[1],r:3,fill:r.status==='stable'?'#16815e':'#999',opacity:.75});c.onclick=()=>el('spaceDetail').textContent=JSON.stringify(r,null,2);s.appendChild(c);});
 let label=node('text',{x:15,y:22,'font-size':13});label.textContent=mode+' · '+rs.length+' 点 · '+(1/scale).toFixed(3)+' mm/绘图单位';s.appendChild(label);
 el('spaceDetail').textContent=JSON.stringify({points:rs.length,frame:'arm root',bounds_mm:bounds,azimuth_deg:+el('azimuth').value,elevation_deg:+el('elevation').value},null,2);
}
['azimuth','elevation'].forEach(id=>el(id).oninput=spaceDraw);
let drag=null;el('space').onpointerdown=e=>{if(el('projection').value!=='3D')return;drag=[e.clientX,e.clientY,+el('azimuth').value,+el('elevation').value];el('space').setPointerCapture(e.pointerId);};
el('space').onpointermove=e=>{if(!drag)return;el('azimuth').value=Math.max(-180,Math.min(180,drag[2]+(e.clientX-drag[0])*.5));el('elevation').value=Math.max(-90,Math.min(90,drag[3]-(e.clientY-drag[1])*.5));spaceDraw();};
el('space').onpointerup=el('space').onpointercancel=()=>drag=null;
el('saveSpace').onclick=()=>{let copy=el('space').cloneNode(true);copy.setAttribute('xmlns',ns);let url=URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(copy)],{type:'image/svg+xml'})),a=document.createElement('a');a.href=url;a.download='workspace-'+el('arm').value+'-'+el('projection').value+'.svg';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
'''
