#!/usr/bin/env python3
"""Render an existing guard study; no planning or manual sketch input."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
import test_remote_control
from rexmi_bt_tests.nav import planning_snapshot


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args()
    r=json.loads((args.directory/'result.json').read_text());g,_=planning_snapshot.load(r['source'])
    selected=r.get('selected');first=r['stages'][0].get('best')
    norm=Normalize(1,20);cmap=plt.get_cmap('plasma');rgba=cmap(norm(g.cost.T))
    rgba[g.blocked.T]=[.22,.24,.27,1];rgba[~g.known.T]=[.85,.85,.85,1]
    x0,y0=np.asarray(g.origin)-g.world_size/2
    xs=x0+(np.arange(g.n_cells)+.5)*g.cell_size;ys=y0+(np.arange(g.n_cells)+.5)*g.cell_size
    routes=[v['best']['route'] for v in r['stages'] if v.get('best')]
    if selected:routes.append(selected['route'])
    allpoints=np.vstack(routes)
    bounds=(min(-6.,allpoints[:,0].min()-1),max(8.,allpoints[:,0].max()+1),
            min(-2.,allpoints[:,1].min()-1),max(15.,allpoints[:,1].max()+1))
    def terrain(ax):
        ax.imshow(rgba,origin='lower',extent=[x0,x0+g.world_size,y0,y0+g.world_size])
        cs=ax.contour(xs,ys,g.surface.T,levels=[1,2,3,4,5],colors='white',linewidths=.5,alpha=.65)
        ax.clabel(cs,fmt='%g m',fontsize=8)
        ax.set(xlim=bounds[:2],ylim=bounds[2:],xlabel='World X (m)',ylabel='World Y (m)')
        for point,label in ((r['start'],'START'),(r['goal'],'GOAL')):
            ax.scatter(*point,c='white',edgecolor='black',s=22,zorder=8)
            ax.annotate(label,point,xytext=(4,6),textcoords='offset points',color='white',weight='bold',fontsize=8)
    fig,axes=plt.subplots(2,3,figsize=(16,12))
    for ax,row in zip(axes.flat,r['stages']):
        terrain(ax);best=row.get('best')
        if best:
            path=np.asarray(best['route']);ax.plot(*path.T,color='#ff762c',lw=2.4)
            violations=best['violations'];seen=set()
            for v in violations:
                pos=tuple(v.get('point',v.get('start')))
                if pos in seen:continue
                seen.add(pos);ax.scatter(*pos,s=30,marker='x',color='#ff3b3b',zorder=9)
            suffix=f"score {best['rank'][0]:.2f} | {best['rank'][1]:.1f} m | {len(violations)} remaining violations"
        else:suffix='No complete witness'
        ax.set_title(f"{row['level']}. {row['name']}\n{suffix}",fontsize=10)
    fig.suptitle('Same terrain, same objective — constraints added cumulatively',fontsize=17,weight='bold')
    fig.text(.5,.02,'Orange: best recorded witness at each stage (not certified optimal). Red crosses: checks still violated.\n'
        'Gray: original planning exclusions, progressively enforced by the ladder. No physical trial.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.06,1,.95]);fig.savefig(args.directory/'guard_stages.png',dpi=150);plt.close(fig)
    fig=plt.figure(figsize=(15,12));gs=fig.add_gridspec(3,2,height_ratios=[3.5,1,1],hspace=.4)
    ax=fig.add_subplot(gs[0,0]);terrain(ax)
    bx=fig.add_subplot(gs[0,1],projection='3d',computed_zorder=False)
    xi=np.flatnonzero((xs>=bounds[0])&(xs<=bounds[1]));yi=np.flatnonzero((ys>=bounds[2])&(ys<=bounds[3]))
    X,Y=np.meshgrid(xs[xi],ys[yi]);Z=g.surface[np.ix_(xi,yi)].T
    bx.plot_surface(X,Y,Z,facecolors=rgba[np.ix_(yi,xi)],linewidth=0,antialiased=False,
        shade=False,rcount=len(yi),ccount=len(xi),zorder=1)
    height=fig.add_subplot(gs[1,:])
    for witness,color,label,style in ((first,'#19d5ed','Relaxed reference','--'),
                                     (selected,'#ff762c','Score-selected, full static guards','-'),
                                     (r.get('legacy_inventory_control',{}).get('best'),'#dddddd','Shorter alternative, same guards','-')):
        if not witness:continue
        path=np.asarray(witness['route']);ax.plot(*path.T,color=color,lw=2.7,ls=style,label=label)
        dense=np.concatenate([np.linspace(a,b,max(2,int(np.linalg.norm(b-a)/.1)+1)) for a,b in zip(path,path[1:])])
        z=[g.surface[g.world_to_cell(*v)]+.12 for v in dense]
        bx.plot(*dense.T,z,color=color,lw=2.5,ls=style,zorder=5)
        profile=witness['metrics']['height_profile']
        height.plot([v['distance_m'] for v in profile],[v['height_m'] for v in profile],color=color,lw=2,label=label,ls=style)
    ax.legend(loc='lower left',fontsize=9);ax.set_title('Costs + height contours')
    bx.view_init(elev=35,azim=-65);bx.set(xlabel='World X (m)',ylabel='World Y (m)',zlabel='Height (m)',title='Same costs draped over terrain')
    height.set(xlabel='Distance along route (m)',ylabel='Footprint mean height (m)');height.grid(alpha=.25);height.legend()
    tableax=fig.add_subplot(gs[2,:]);tableax.axis('off')
    rows=[]
    for row in r['stages']:
        best=row.get('best');rows.append([row['name'],'Yes' if best else 'No',
            f"{best['rank'][0]:.2f}" if best else '—',f"{best['rank'][1]:.1f}" if best else '—',
            str(len(best['violations'])) if best else '—'])
    legacy=r.get('legacy_inventory_control',{}).get('best')
    if legacy:
        rows.append(['Original candidates, all static guards','Yes',f"{legacy['rank'][0]:.2f}",
                     f"{legacy['rank'][1]:.1f}",str(len(legacy['violations']))])
    table=tableax.table(cellText=rows,colLabels=['Cumulative guards','Route found','Sustained score','Distance (m)','Remaining violations'],loc='center',cellLoc='center',colWidths=[.4,.14,.16,.14,.16])
    table.auto_set_font_size(False);table.set_fontsize(9);table.scale(1,1.5)
    fig.suptitle('Guard ablation — complete routes retained',fontsize=18,weight='bold')
    fig.subplots_adjust(left=.07,right=.95,top=.92,bottom=.17)
    cax=fig.add_axes([.22,.10,.56,.015]);fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation='horizontal',ticks=[1,5,10,15,20])
    cax.set_xlabel('Unchanged terrain cost: purple low → yellow high. Gray = original planning exclusion.',fontsize=9)
    status=(f"Selected stage {r['selected_level']}; score {selected['rank'][0]:.2f}; {selected['rank'][1]:.1f} m. "
        +('Sampled-graph optimum certified.' if selected['optimal'] else 'Best found; optimum not certified.')) if selected else 'No selected route.'
    fig.text(.5,.025,status+'\nMap-only geometry. Traction, stopping and physical ascent remain unproven. Runtime unchanged.',ha='center',fontsize=10)
    fig.savefig(args.directory/'route_comparison.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
