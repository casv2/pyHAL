from ase.io import read, write
import matplotlib.pyplot as plt
from ase.units import fs
from copy import deepcopy
import torchani
import numpy as np
import os
import sys

def get_comm_energy_forces(ani2x, at):
    Es = []
    Fs = []

    for i in range(8):
        calc = ani2x[i].ase()
        at.set_calculator(calc)
        Es.append(at.get_potential_energy())
        Fs.append(at.get_forces())

    meanE = np.mean(Es)
    meanF = np.mean(Fs, axis=0)

    varE = np.sum([ (Es[i] - meanE)**2 for i in range(8)])/8
    varF = np.sum([2 * (Es[i] - meanE) * (Fs[i] - meanF) for i in range(8)], axis=0)/8

    F_RMSE = np.sqrt(np.max([np.power(meanF - Fs[i], 2) for i in range(8)]))

    return meanE, varE, meanF, varF, F_RMSE

def velo_verlet_com(at, ani2x, dt, tau, minF):
    meanE, varE, meanF, varF, F_RMSE = get_comm_energy_forces(ani2x, at)
    forces = meanF - tau * varF

    p = at.get_momenta()
    p += 0.5 * dt * forces
    masses = at.get_masses()[:, np.newaxis]
    r = at.get_positions()

    at.set_positions(r + dt * p / masses)
    at.set_momenta(p, apply_constraint=False)

    meanE, varE, meanF, varF, F_RMSE = get_comm_energy_forces(ani2x, at)
    forces = meanF - tau * varF

    at.set_momenta(at.get_momenta() + 0.5 * dt * forces)

    p = np.max(np.linalg.norm(varF, axis=1) / (np.linalg.norm(meanF, axis=1) + minF))

    return at, p, F_RMSE, varE

def HAL(at, nsteps=1000, tau=0.01, dtau=0.1, ntau=100, minF=0.1, pmax=0.3, dt=0.5):
    Ps = []
    varEs = []
    Es = []
    F_RMSEs = []
    al = []

    ani2x = torchani.models.ANI2x()

    i = 0
    running = True
    while running and i < nsteps:
        print(i, tau)
        at, p, F_RMSE, varE = velo_verlet_com(at, ani2x, dt * fs, tau, minF)
        Ps.append(p)
        varEs.append(varE)
        F_RMSEs.append(F_RMSE)
        Es.append(at.get_potential_energy())
        #al.append(deepcopy(at))
        if i % ntau == 0:
            tau += dtau
        if p > pmax or i+2 > nsteps:
            running = False
            al.append(at)
        i+=1

    #F_RMSE_ind = np.argmax(F_RMSEs)
    #varE_ind = np.argmax(varEs)
    #P_ind = np.argmax(Ps)

    # print("P_ind: {}".format(P_ind))
    # print("varE_ind: {}".format(varE_ind))
    # print("F_RMSE_ind: {}".format(F_RMSE_ind))

    # write("at_P_max.xyz", al[P_ind])
    # write("at_varE_max.xyz", al[varE_ind])
    # write("at_F_RMSE_max.xyz", al[F_RMSE_ind])

    fig, axs = plt.subplots(4, figsize=(6,8))
    axs[0].plot(Es)
    axs[0].set_ylabel("E")
    axs[1].plot(varEs)
    axs[1].set_ylabel("varE")
    axs[2].plot(Ps)
    axs[2].set_ylabel("P")
    axs[3].plot(F_RMSEs)
    axs[3].set_ylabel("F_RMSE")
    axs[3].set_xlabel("MD step")
    plt.savefig("report.pdf")

    write("./selected_config.xyz", al)