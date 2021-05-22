from ase.io import read, write
import matplotlib.pyplot as plt
from ase.units import fs
from ase.units import kB
from copy import deepcopy
from numpy.core.numeric import full_like
import torchani
import numpy as np
import torch
import ase
from copy import deepcopy

def get_comm_energy_forces(ani2x,model, at):
    Fs = []

    for i in range(8):
        calc = ani2x[i].ase()
        at.set_calculator(calc)
        Fs.append(at.get_forces())

    species = torch.tensor([at.get_atomic_numbers()], dtype=torch.long)
    coordinates = torch.tensor([at.get_positions()], requires_grad=True, dtype=torch.float32)

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

def velo_verlet_langevin(at, ani2x, model, dt, T, gamma, tau):
    varE, meanF, varF = get_comm_energy_forces(ani2x,model, at)
    forces = at.get_forces() - tau * varF

    p = at.get_momenta()
    p += 0.5 * dt * forces
    masses = at.get_masses()[:, np.newaxis]
    p = random_p_update(p, masses, gamma, T, dt)
    r = at.get_positions()

    at.set_positions(r + dt * p / masses)
    at.set_momenta(p, apply_constraint=False)

    varE, meanF, varF = get_comm_energy_forces(ani2x,model, at)
    forces = at.get_forces()  - tau * varF

    p = at.get_momenta() + 0.5 * dt * forces
    p = random_p_update(p, masses, gamma, T, dt)
    at.set_momenta(p)

    varEmax = np.max(varE)
    Fu = np.max(np.linalg.norm(varF, axis=1))

    return at, varEmax, Fu

# def random_p_update(p,masses,gamma,kBT,dt):
#     v = p / masses
#     R = np.random.standard_normal(size=(len(masses), 3))
#     c1 = np.exp(-gamma*dt)
#     c2 = np.sqrt(1-c1*c1)*np.sqrt(kBT / masses)
#     v_new = c1*v + (R.T*c2).T
#     return masses * v_new

def random_p_update(p,masses,gamma,kBT,dt):
    v = p / masses
    R = np.random.standard_normal(size=(len(masses), 3))
    c1 = np.exp(-gamma*dt)
    c2 = np.sqrt(1-c1*c1)*np.sqrt(kBT / masses)
    v_new = c1*v + (R* c2)
    return v_new* masses

def HAL(at, fname, nsteps=1000, tau=0.01, dtau=0.1, ntau=100, Fu_max=0.3, dt=0.5, T=0, gamma=0.2, write_files=True):
    Fu_s = np.zeros(nsteps)
    varEs = np.zeros(nsteps)
    Es = np.zeros(nsteps)
    Ts = np.zeros(nsteps)
    al = []

    ani2x = torchani.models.ANI2x()
    model = torchani.models.ANI2x(periodic_table_index=True)

    i = 0
    running = True
    while running and i < nsteps:
        if T == 0:
            at, varEmax, Fu = velo_verlet_com(at, ani2x, model, dt * fs, tau)
        else:
            at, varEmax, Fu = velo_verlet_langevin(at, ani2x, model, dt * fs, T * kB, gamma, tau)
        Fu_s[i] = Fu
        varEs[i] = varEmax
        Es[i] = at.get_potential_energy()
        T = (at.get_kinetic_energy()/len(at)) / (1.5 * kB)
        Ts[i] = T
        print(i, tau, T, Fu)
        if i % ntau == 0:
            tau += dtau
        if Fu > Fu_max:
            at.info["Fu"]= Fu
            at2 = deepcopy(at)
            al.append(at2)
        if T > 1000:
            running= False
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
        fig, axs = plt.subplots(4, figsize=(6,8))
        axs[0].plot(Es[:i])
        axs[0].set_ylabel("E")
        axs[1].plot(varEs[:i])
        axs[1].set_ylabel("max varE")
        axs[2].plot(Fu_s[:i])
        axs[2].set_ylabel("norm Fu")
        axs[3].plot(Ts[:i])
        axs[3].set_ylabel("T [K]")
        plt.tight_layout()
        plt.savefig("report_{}.pdf".format(fname))

        write("./selected_configs_{}.xyz".format(fname), al)
    else:
        return Es[:i], Fu_s[:i]