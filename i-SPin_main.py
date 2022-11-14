import numpy as np
import matplotlib.pyplot as plt
import pyfftw
import random
import kdkfunc
import conservefunc
import rtplotfunc

N = 51  # number of spatial lattice sites (keep odd)
t = 0  # initial time
tend = 20  # time at which simulation ends
L = 5  # Domain [0,L] x [0,L] x [0,L]
alpha = 1  # Coefficient of the |(psi.psi)|^2 interaction term. Set to 1 for vector dark matter
Lambda = -0.01  # self interaction strength; positive means attractive, negative means repulsive
eta = 24  # number of points to sample a full 2 pi rotation

dx = L / N
xspace = np.linspace(0, L, num=N + 1)  # array of x values
xspace = xspace[0:N]  # to erase the last point, as it is the same as the first point in our periodic box
xx, yy, zz = np.meshgrid(xspace, xspace, xspace, sparse=True)

switch = 0  # switch between gravity (=0) for cosmological purposes, and trapping potentials (=1) relevant for BECs in lab.
# omegas for the harmonic trap (for BEC systems)
omegax = 3
omegay = 3
omegaz = 3

# initial field configuration: Gaussian in every component with random phase; modify as desired
sigma = 3.0 * dx
G1 = np.exp(
    -((xx - random.uniform(3 * dx, L - 3 * dx)) ** 2 + (yy - random.uniform(3 * dx, L - 3 * dx)) ** 2 + (
            zz - random.uniform(3 * dx - L/2, L/2 - 3 * dx)) ** 2) / (
            2 * sigma ** 2))
G2 = np.exp(
    -((xx - random.uniform(3 * dx, L - 3 * dx)) ** 2 + (yy - random.uniform(3 * dx, L - 3 * dx)) ** 2 + (
            zz - random.uniform(3 * dx - L/2, L/2 - 3 * dx)) ** 2) / (
            2 * sigma ** 2))
G3 = np.exp(
    -((xx - random.uniform(3 * dx, L - 3 * dx)) ** 2 + (yy - random.uniform(3 * dx, L - 3 * dx)) ** 2 + (
            zz - random.uniform(3 * dx - L/2, L/2 - 3 * dx)) ** 2) / (
            2 * sigma ** 2))
psi1 = np.exp(1.j * random.uniform(0, 2 * np.pi))*G1 + np.exp(1.j * random.uniform(0, 2 * np.pi))*G2 + np.exp(1.j * random.uniform(0, 2 * np.pi))*G3
psi2 = np.exp(1.j * random.uniform(0, 2 * np.pi))*G1 + np.exp(1.j * random.uniform(0, 2 * np.pi))*G2 + np.exp(1.j * random.uniform(0, 2 * np.pi))*G3
psi3 = np.exp(1.j * random.uniform(0, 2 * np.pi))*G1 + np.exp(1.j * random.uniform(0, 2 * np.pi))*G2 + np.exp(1.j * random.uniform(0, 2 * np.pi))*G3

fullrot = 2 * np.pi * dx ** 2 * (1 / 3)  # one full rotation of the laplacian phase
dt = (1 / eta) * fullrot  # timestep
print("dt = ", dt)
Nt = int(np.ceil(tend / dt))  # number of time iterations

# creating k-space and the associated laplacian matrix
nvalues = np.arange(-(N - 1) / 2, (N + 1) / 2)
kspace = 2.0 * N / L * np.sin(np.pi * nvalues / N)
kx, ky, kz = np.meshgrid(kspace, kspace, kspace, sparse=True)
kx = pyfftw.interfaces.numpy_fft.ifftshift(kx)
ky = pyfftw.interfaces.numpy_fft.ifftshift(ky)
kz = pyfftw.interfaces.numpy_fft.ifftshift(kz)
ksq = kx ** 2 + ky ** 2 + kz ** 2

# preparing handles for figures
plotrealtime = True  # switch on for plotting as the simulation goes along
tdraw = 5 * dt  # draw frequency for plotting in real time
plotcount = 1  # initial count for plotting in real time
fig = plt.figure(figsize=(9, 7), dpi=80)
grid = plt.GridSpec(2, 2, wspace=0.6, hspace=0.3)
pl1 = plt.subplot(grid[0, 0])
pl2 = plt.subplot(grid[0, 1])
pl3 = plt.subplot(grid[1, 0])
pl4 = plt.subplot(grid[1, 1])

# creating arrays for tracking them as simulation goes along
numfrac = []
spinfrac = []
rtime = []

# initial total mass and spin
numini = np.sum(np.abs(psi1) ** 2 + np.abs(psi2) ** 2 + np.abs(psi3) ** 2) * dx ** 3
print("Total mass = ", numini)
spinxini = np.real((1.j * np.sum(psi2 * np.conjugate(psi3)) - 1.j * np.sum(psi3 * np.conjugate(psi2))) * dx ** 3)
spinyini = np.real((-1.j * np.sum(psi1 * np.conjugate(psi3)) + 1.j * np.sum(psi3 * np.conjugate(psi1))) * dx ** 3)
spinzini = np.real((1.j * np.sum(psi1 * np.conjugate(psi2)) - 1.j * np.sum(psi2 * np.conjugate(psi1))) * dx ** 3)
print("Total spin = ", [spinxini, spinyini, spinzini])

for i in range(Nt):

    # Performing drift/2 + kick + drift/2 (main part of the algorithm)
    psi1, psi2, psi3, rho, spindensity = kdkfunc.kdkevolve(L, dt, alpha, Lambda, eta, psi1, psi2, psi3, ksq, switch, omegax, omegay, omegaz, xx, yy, zz)

    t += dt

    # updating arrays of quantities
    rtime, numfrac, spinfrac = conservefunc.dfraceval(t, dx, rtime, psi1, psi2, psi3, numini, spinxini, spinyini, spinzini, numfrac, spinfrac)

    # plot in real time
    plotthisturn = False
    if t + dt > plotcount * tdraw:
        plotthisturn = True
    if (plotrealtime and plotthisturn) or (i == Nt - 1):

        rtplotfunc.plotting(N, pl1, pl2, pl3, pl4, rho, spindensity, numfrac, spinfrac, rtime)

        plt.pause(0.00001)
        plotcount += 1
