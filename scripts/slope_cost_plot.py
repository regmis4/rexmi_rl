"""Shared accepted cost-map colors and normalization for fit and run reports."""
import numpy as np

COLORS = dict(any_exit='#00d9ff', matched_manual_exit='#ff9c24', manual='#72f23b', rejected='#ff6363')


def draw(ax, model):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    g=model.grid;p=model.parameters
    # Identical steepest-uphill scalar view used in the accepted route figure.
    penalty=p.uphill_weight*np.expm1(np.maximum(model.terrain_angle-p.easy_angle_deg,0.)/p.exponential_scale_deg)
    density=g.cost+(1+p.terrain_weight)*penalty*(1+p.roughness_interaction*model.roughness)
    norm=LogNorm(vmin=2.,vmax=max(20.,float(np.nanpercentile(density[g.known&~g.blocked],99))))
    rgba=plt.cm.magma(norm(density.T))
    rgba[g.blocked.T]=(.3,.31,.33,1.);rgba[~g.known.T]=(.12,.13,.15,1.)
    lower=np.asarray(g.origin)-g.world_size/2
    ax.imshow(rgba,origin='lower',extent=[lower[0],lower[0]+g.world_size,lower[1],lower[1]+g.world_size])
    x=lower[0]+(np.arange(g.n_cells)+.5)*g.cell_size
    y=lower[1]+(np.arange(g.n_cells)+.5)*g.cell_size
    contours=ax.contour(x,y,np.where(g.known,g.surface,np.nan).T,levels=np.arange(0,6,.5),colors='white',alpha=.4,linewidths=.7)
    ax.clabel(contours,levels=np.arange(1,5),fontsize=8,fmt='%.0f m')
    return plt.cm.ScalarMappable(norm=norm,cmap='magma')
