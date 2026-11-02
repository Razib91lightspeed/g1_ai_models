#!/usr/bin/env python3
"""Record real MuJoCo browser telemetry and draw 2D actual route (stdlib only).
Simulation monitor: does not command or control robot.
"""
import argparse, csv, html, json, math, time
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError

TERMINAL={'finished','collision','fallen','blocked','turn_blocked','turn_required','turn_timeout','home_heading_unverified','assistance_required'}
FIELDS=['wall_time','simulation_time_s','mission','status','x_m','y_m','heading_deg','return_mode','return_checkpoint_index','geofence_ok','obstacles_spawned','contact_with_obstacle']

def draw(rows, filename, lab_length, lab_width, margin, goal):
    W,H=1120,660; P=65
    xmin,xmax=-0.5,lab_length+0.5; ymin,ymax=-lab_width/2-0.5,lab_width/2+0.5
    scale=min((W-2*P)/(xmax-xmin),(H-2*P)/(ymax-ymin))
    ox=(W-scale*(xmax-xmin))/2; oy=(H-scale*(ymax-ymin))/2
    X=lambda x: ox+(x-xmin)*scale
    Y=lambda y: H-oy-(y-ymin)*scale
    def poly(phase,color):
        segments=[]; cur=[]
        for r in rows:
            if r['mission']==phase:
                cur.append(f'{X(float(r["x_m"])):.1f},{Y(float(r["y_m"])):.1f}')
            elif cur:
                segments.append(cur);cur=[]
        if cur: segments.append(cur)
        return ''.join(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{" ".join(seg)}" />' for seg in segments if len(seg)>1)
    def txt(x,y,s,size=14,fill='#dfe7ee'):
        return f'<text x="{x}" y="{y}" fill="{fill}" font-size="{size}" font-family="sans-serif">{html.escape(str(s))}</text>'
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',f'<rect width="{W}" height="{H}" fill="#0e1825"/>']
    parts.append(f'<rect x="{X(0):.1f}" y="{Y(lab_width/2):.1f}" width="{lab_length*scale:.1f}" height="{lab_width*scale:.1f}" fill="#253646" stroke="#8294a8" stroke-width="2"/>')
    parts.append(f'<rect x="{X(margin):.1f}" y="{Y(lab_width/2-margin):.1f}" width="{(lab_length-2*margin)*scale:.1f}" height="{(lab_width-2*margin)*scale:.1f}" fill="none" stroke="#fbbf24" stroke-dasharray="8 6" stroke-width="2"/>')
    for k in range(0, int(lab_length)+1,2):
        parts.append(f'<line x1="{X(k)}" y1="{Y(-lab_width/2)}" x2="{X(k)}" y2="{Y(lab_width/2)}" stroke="#536577" stroke-opacity=".4"/>')
        parts.append(txt(X(k)-7,Y(-lab_width/2)+25,str(k),12))
    parts += [poly('outbound','#40dc9a'),poly('returning','#56b7ff')]
    for px,py,name,color in [(0,0,'HOME','#e2e8f0'),(goal[0],goal[1],'GOAL','#ffe082')]:
        parts.append(f'<circle cx="{X(px)}" cy="{Y(py)}" r="7" fill="{color}"/>')
        parts.append(txt(X(px)+9,Y(py)-10,name,14,color))
    if rows:
        r=rows[-1]; px,py=float(r['x_m']),float(r['y_m'])
        parts.append(f'<circle cx="{X(px)}" cy="{Y(py)}" r="8" fill="#ff6678" stroke="white" stroke-width="1.5"/>')
        parts.append(txt(70,30,f'Latest: {r["status"]} | {r["mission"]} | ({px:.2f},{py:.2f}) m | Return: {r["return_mode"]}'))
    parts.append(txt(70,H-19,'GREEN: actual outbound   BLUE: actual return   YELLOW DASH: provisional geofence',14))
    parts.append('</svg>')
    filename.write_text('\n'.join(parts),encoding='utf-8')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--url',default='http://127.0.0.1:8771/status')
    ap.add_argument('--out',default='g1_route_capture')
    ap.add_argument('--interval',type=float,default=0.25)
    ap.add_argument('--max-seconds',type=float,default=240)
    ap.add_argument('--goal-x',type=float,default=7.0)
    ap.add_argument('--goal-y',type=float,default=-1.0)
    args=ap.parse_args()
    prefix=Path(args.out); prefix.parent.mkdir(parents=True,exist_ok=True)
    rows=[]; last_sim=-float('inf'); started=time.monotonic(); terminal_since=None
    csvfile=prefix.with_suffix('.csv'); svgfile=prefix.with_suffix('.svg')
    with csvfile.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader()
        print(f'Recording live trajectory to {csvfile}; map {svgfile}',flush=True)
        while time.monotonic()-started<args.max_seconds:
            try:
                with urlopen(args.url,timeout=3) as response: d=json.load(response)
                t=float(d['simulation_time_s'])
                if t<last_sim: # Restart pressed in browser: start a clean plot
                    rows=[]; f.seek(0);f.truncate();writer.writeheader()
                    print('Browser Restart detected: cleared previous run',flush=True)
                if t!=last_sim or not rows:
                    row={key:d.get(key,'') for key in FIELDS};row['wall_time']=time.strftime('%Y-%m-%dT%H:%M:%S')
                    if row['x_m'] not in ('',None) and row['y_m'] not in ('',None):
                        writer.writerow(row);f.flush();rows.append(row)
                        if len(rows)%8==0 or d.get('status') in TERMINAL:
                            draw(rows,svgfile,float(d.get('lab_length_m',12)),float(d.get('lab_width_m',6)),float(d.get('wall_margin_m',0.5)),(args.goal_x,args.goal_y))
                last_sim=t
                if d.get('status') in TERMINAL:
                    if terminal_since is None:terminal_since=time.monotonic()
                    if time.monotonic()-terminal_since>=2:break
                else:terminal_since=None
            except (URLError,KeyError,ValueError,TimeoutError) as e:
                print('Waiting for live browser status:',e,flush=True)
            time.sleep(args.interval)
    if rows:
        d=locals().get('d',{})
        draw(rows,svgfile,float(d.get('lab_length_m',12)),float(d.get('lab_width_m',6)),float(d.get('wall_margin_m',0.5)),(args.goal_x,args.goal_y))
    print('Saved:',csvfile,svgfile,flush=True)
if __name__=='__main__': main()
