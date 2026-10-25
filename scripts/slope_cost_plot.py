"""Shared accepted cost-map colors and normalization for fit and run reports."""
import numpy as np

COLORS = dict(any_exit='#00d9ff', matched_manual_exit='#ff9c24', manual='#72f23b', rejected='#ff6363')


def draw(ax, model):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    # Use the model's package so both simulator and CPU report imports work.
    from importlib import import_module
    view=import_module(type(model).__module__.rsplit('.',1)[0]+'.terrain_display').from_model(model)
    g=model.grid
    norm=LogNorm(vmin=view['vmin'],vmax=view['vmax'])
    rgba=view['rgb'].transpose(1,0,2)
    lower=np.asarray(g.origin)-g.world_size/2
    ax.imshow(rgba,origin='lower',extent=[lower[0],lower[0]+g.world_size,lower[1],lower[1]+g.world_size])
    x=lower[0]+(np.arange(g.n_cells)+.5)*g.cell_size
    y=lower[1]+(np.arange(g.n_cells)+.5)*g.cell_size
    contours=ax.contour(x,y,np.where(g.known,g.surface,np.nan).T,levels=np.arange(0,6,.5),colors='white',alpha=.4,linewidths=.7)
    ax.clabel(contours,levels=np.arange(1,5),fontsize=8,fmt='%.0f m')
    return plt.cm.ScalarMappable(norm=norm,cmap='magma')
