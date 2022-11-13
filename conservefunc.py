# This is the module to create the arrays for fractional changes in total mass and spin

import numpy as np

def dfraceval(t, dx, rtime, psi1, psi2, psi3, numini, spinxini, spinyini, spinzini, numfrac, spinfrac):

    rtime.append(t)
    numfrac.append(
        np.abs(np.sum(np.abs(psi1) ** 2 + np.abs(psi2) ** 2 + np.abs(psi3) ** 2) * dx ** 3 - numini) / numini)
    spindiffxt = np.abs(
        (1.j * np.sum(psi2 * np.conjugate(psi3)) - 1.j * np.sum(psi3 * np.conjugate(psi2))) * dx ** 3 - spinxini)
    spindiffyt = np.abs(
        (-1.j * np.sum(psi1 * np.conjugate(psi3)) + 1.j * np.sum(psi3 * np.conjugate(psi1))) * dx ** 3 - spinyini)
    spindiffzt = np.abs(
        (1.j * np.sum(psi1 * np.conjugate(psi2)) - 1.j * np.sum(psi2 * np.conjugate(psi1))) * dx ** 3 - spinzini)
    spindifft = np.sqrt(spindiffxt ** 2 + spindiffyt ** 2 + spindiffzt ** 2)
    spinini = np.sqrt(spinxini ** 2 + spinyini ** 2 + spinzini ** 2)
    spinfrac.append(spindifft / spinini)

    return rtime, numfrac, spinfrac
