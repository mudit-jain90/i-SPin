# This is the plotting module

import numpy as np
import matplotlib.pyplot as plt

def plotting(N, pl1, pl2, pl3, pl4, rho, spindensity, numfrac, spinfrac, rtime):

    plt.sca(pl1)
    plt.cla()
    plt.imshow(np.log10(np.sum(rho, axis=2) / N), cmap='afmhot')
    plt.clim(-5, 2)
    plt.gca().invert_yaxis()
    pl1.get_xaxis().set_visible(False)
    pl1.get_yaxis().set_visible(False)
    pl1.set_aspect('equal')
    plt.title(r'Mass density projection')

    plt.sca(pl2)
    plt.cla()
    plt.imshow(np.log10(np.sum(spindensity, axis=2) / N), cmap='magma')
    plt.clim(-5, 2)
    plt.gca().invert_yaxis()
    pl2.get_xaxis().set_visible(False)
    pl2.get_yaxis().set_visible(False)
    pl2.set_aspect('equal')
    plt.title(r'Spin density projection')

    plt.sca(pl3)
    plt.cla()
    plt.plot(rtime, numfrac)
    plt.yscale("log")
    plt.xscale("linear")
    plt.title(r'$\Delta_N$ vs time')

    plt.sca(pl4)
    plt.cla()
    plt.plot(rtime, spinfrac)
    plt.yscale("log")
    plt.xscale("linear")
    plt.title(r'$\Delta_S$ vs time')
