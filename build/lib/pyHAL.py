from ase.io import read, write
import matplotlib.pyplot as plt
from ase.units import fs
from copy import deepcopy
import torchani
import numpy as np
import torch
import ase

def get_comm_energy_forces(ani2x,model, at):
    Fs = []

    for i in range(8):
        calc = ani2x[i].ase()
        at.set_calculator(calc)
        Fs.append(at.get_forces())

    species = torch.tensor([at.get_atomic_numbers()], dtype=torch.long)
    coordinates = torch.tensor([at.get_positions()], requires_grad=True, dtype=torch.float32)

    #energy = model((species, coordinates)).energies

    _, atomic_energies = model.atomic_energies((species, coordinates), average=False)

    Es = atomic_energies.detach().numpy() * ase.units.Hartree

    meanE = np.mean(Es, axis=0)
    meanF = np.mean(Fs, axis=0)

    varE = np.var(Es, axis=0)
    varF = np.sum([2 * (Es[i] - meanE) * (Fs[i] - meanF).T for i in range(8)], axis=0)/8

    return varE, meanF, varF.T

def velo_verlet_com(at, ani2x,model,  dt, tau):
    varE, meanF, varF = get_comm_energy_forces(ani2x,model, at)
    forces = meanF - tau * varF

    p = at.get_momenta()
    p += 0.5 * dt * forces
    masses = at.get_masses()[:, np.newaxis]
    r = at.get_positions()

    at.set_positions(r + dt * p / masses)
    at.set_momenta(p, apply_constraint=False)

    varE, meanF, varF = get_comm_energy_forces(ani2x,model, at)
    forces = meanF - tau * varF

    at.set_momenta(at.get_momenta() + 0.5 * dt * forces)

    varEmax = np.max(varE)
    Fu = np.max(np.linalg.norm(varF, axis=1))

    return at, varEmax, Fu

def HAL(at, fname, nsteps=1000, tau=0.01, dtau=0.1, ntau=100, Fu_max=0.3, dt=0.5, write_files=True):
    Fu_s = []
    varEs = []
    Es = []
    al = []

    ani2x = torchani.models.ANI2x()
    model = torchani.models.ANI2x(periodic_table_index=True)

    i = 0
    running = True
    while running and i < nsteps:
        print(i, tau)
        at, varEmax, Fu = velo_verlet_com(at, ani2x,model, dt * fs, tau)
        Fu_s.append(Fu)
        varEs.append(varEmax)
        Es.append(at.get_potential_energy())
        if i % ntau == 0:
            tau += dtau
        if Fu > Fu_max or i+2 > nsteps:
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

    if write_files:
        fig, axs = plt.subplots(3, figsize=(6,8))
        axs[0].plot(Es)
        axs[0].set_ylabel("E")
        axs[1].plot(varEs)
        axs[1].set_ylabel("max varE")
        axs[2].plot(Fu_s)
        axs[2].set_ylabel("norm Fu")
        plt.tight_layout()
        plt.savefig("report_{}.pdf".format(fname))

        write("./selected_config_{}.xyz".format(fname), al)
    else:
        return Es, Fu_s