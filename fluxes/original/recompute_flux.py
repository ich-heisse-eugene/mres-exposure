#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

stars, vmag = np.loadtxt("input.lst", unpack=True, delimiter=";", dtype="str")

for i in range(len(stars)):
    wl, fl = np.loadtxt(stars[i].strip()+"_fl.dat", unpack=True, dtype=float, comments='#')
    wl *= 10.
    if stars[i].find('hd') != -1:
        fl = 0.503 * wl * fl
    elif stars[i].find('hr') != -1:
        fl = 5.03 * wl * fl
    fl *= 10**(0.4 * float(vmag[i]))
    np.savetxt(stars[i].strip()+"_sc_fl.dat", np.vstack((wl, fl)).transpose(), fmt="%.1f")
    plt.plot(wl, fl, ls='-', lw=0.5, label=stars[i].strip())

plt.legend()
plt.show()
