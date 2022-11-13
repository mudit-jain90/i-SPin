# This is the main module that performs half-drift + kick + half-drift

import numpy as np
import pyfftw

def kdkevolve(L, dt, alpha, Lambda, eta, psi1, psi2, psi3, ksq, switch, omegax, omegay, omegaz, xx, yy, zz):

    #  Performing half drift
    psi1 = pyfftw.interfaces.numpy_fft.ifftn(
        np.exp(dt * (-1.j * ksq / 4.)) * pyfftw.interfaces.numpy_fft.fftn(psi1))
    psi2 = pyfftw.interfaces.numpy_fft.ifftn(
        np.exp(dt * (-1.j * ksq / 4.)) * pyfftw.interfaces.numpy_fft.fftn(psi2))
    psi3 = pyfftw.interfaces.numpy_fft.ifftn(
        np.exp(dt * (-1.j * ksq / 4.)) * pyfftw.interfaces.numpy_fft.fftn(psi3))

    # Calculating quantities for the full kick evolution
    rho = np.abs(psi1) ** 2 + np.abs(psi2) ** 2 + np.abs(psi3) ** 2
    fsq = psi1 ** 2 + psi2 ** 2 + psi3 ** 2
    magfsq = np.abs(fsq)
    magfsq = np.where(magfsq > rho, rho, magfsq)  # getting rid of possible error due to machine precision
    phase = (np.angle(fsq) + 2.0 * np.pi) % (2.0 * np.pi)
    spindensity = np.sqrt(rho ** 2 - magfsq ** 2)
    U11 = np.cos(dt * alpha * Lambda * rho) * np.cos(dt * alpha * Lambda * spindensity) + dt * alpha * Lambda * np.sinc(
        dt * alpha * Lambda * spindensity / np.pi) * (
                  rho * np.sin(dt * alpha * Lambda * rho) - magfsq * np.sin(phase + dt * alpha * Lambda * rho))
    U12 = -np.sin(dt * alpha * Lambda * rho) * np.cos(dt * alpha * Lambda * spindensity) + dt * alpha * Lambda * np.sinc(
        dt * alpha * Lambda * spindensity / np.pi) * (
                  rho * np.cos(dt * alpha * Lambda * rho) + magfsq * np.cos(phase + dt * alpha * Lambda * rho))
    U21 = np.sin(dt * alpha * Lambda * rho) * np.cos(dt * alpha * Lambda * spindensity) + dt * alpha * Lambda * np.sinc(
        dt * alpha * Lambda * spindensity / np.pi) * (
                  -rho * np.cos(dt * alpha * Lambda * rho) + magfsq * np.cos(phase + dt * alpha * Lambda * rho))
    U22 = np.cos(dt * alpha * Lambda * rho) * np.cos(dt * alpha * Lambda * spindensity) + dt * alpha * Lambda * np.sinc(
        dt * alpha * Lambda * spindensity / np.pi) * (
                  rho * np.sin(dt * alpha * Lambda * rho) + magfsq * np.sin(phase + dt * alpha * Lambda * rho))

    if switch == 0:
        phi = - np.real(
            pyfftw.interfaces.numpy_fft.ifftn(pyfftw.interfaces.numpy_fft.fftn(0.5 * (rho - np.mean(rho))) / (
                    ksq + (ksq == 0))))
        # check CFL condition
        if dt > 2 * np.pi * eta * np.min(np.minimum(np.abs(1 / phi), np.abs(1 / (2 * alpha * Lambda * rho)))):
            print('Caution, CFL condition is invoked. High dense regions are appearing')

    elif switch == 1:
        phi = 0.5 * (omegax ** 2 * (xx - L/2) ** 2 + omegay ** 2 * (yy - L/2) ** 2 + omegaz ** 2 * (zz - L/2) ** 2)

    # Performing full kick
    psi1 = np.exp(-1.j * dt * (phi - 2.0 * Lambda * rho)) * (
            (U11 + 1.j * U21) * (np.real(psi1)) + (U12 + 1.j * U22) * np.imag(
        psi1))
    psi2 = np.exp(-1.j * dt * (phi - 2.0 * Lambda * rho)) * (
            (U11 + 1.j * U21) * (np.real(psi2)) + (U12 + 1.j * U22) * np.imag(
        psi2))
    psi3 = np.exp(-1.j * dt * (phi - 2.0 * Lambda * rho)) * (
            (U11 + 1.j * U21) * (np.real(psi3)) + (U12 + 1.j * U22) * np.imag(
        psi3))

    # Performing half drift
    psi1 = pyfftw.interfaces.numpy_fft.ifftn(
        np.exp(dt * (-1.j * ksq / 4.)) * pyfftw.interfaces.numpy_fft.fftn(psi1))
    psi2 = pyfftw.interfaces.numpy_fft.ifftn(
        np.exp(dt * (-1.j * ksq / 4.)) * pyfftw.interfaces.numpy_fft.fftn(psi2))
    psi3 = pyfftw.interfaces.numpy_fft.ifftn(
        np.exp(dt * (-1.j * ksq / 4.)) * pyfftw.interfaces.numpy_fft.fftn(psi3))

    return psi1, psi2, psi3, rho, spindensity