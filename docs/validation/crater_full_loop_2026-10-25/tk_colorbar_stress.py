import sys,time,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'tests'))
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from types import SimpleNamespace
from threading import Lock
from test_dashboard_live import accepted_model,Dashboard,terrain_display
model,result=accepted_model()
state=dict(pose=SimpleNamespace(x=0,y=0,z=1,yaw=0),cmd=(.4,0,0),
    planned_path=result['routes']['any_exit'],terrain_display=terrain_display.from_model(model),
    travel_profile=[[0,1,0],[1,1.1,10]],speed=.3,actual_yaw_rate=0.)
dashboard=Dashboard(state,Lock(),model.grid,[])
dashboard._create_figure(plt);dashboard._redraw(plt)
errors=[]
dashboard._fig.canvas.manager.window.report_callback_exception=lambda *e: errors.append(str(e))
locator=dashboard._ax_colorbar.get_axes_locator();bar=dashboard._colorbar
started=time.monotonic()
for n in range(1100):
    state['terrain_display']['vmax']=20+n
    dashboard._layer='slope' if n%11==0 else 'terrain'
    dashboard._draw_legend(state['terrain_display'])
    dashboard._fig.canvas.draw_idle();dashboard._fig.canvas.flush_events()
    assert dashboard._colorbar is bar and dashboard._ax_colorbar.get_axes_locator() is locator
    if n%250==0:print('Tk frames',n,flush=True)
assert not errors,errors
dashboard._fig.savefig('outputs/full_loop_2026-10-25/tk_dashboard.png')
plt.close(dashboard._fig)
result=dict(backend=matplotlib.get_backend(),frames=1100,callback_errors=errors,
    seconds=time.monotonic()-started,locator_reused=True)
Path('outputs/full_loop_2026-10-25/tk_stress.json').write_text(json.dumps(result,indent=2)+'\n')
print(result,flush=True)
