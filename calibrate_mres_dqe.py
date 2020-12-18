#!/usr/bin/env python3

from sys import argv, exit
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib as mpl
import spectres

fontsize = 14
mpl.rcParams['xtick.labelsize'] = fontsize
mpl.rcParams['ytick.labelsize'] = fontsize
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'cm'
mpl.rcParams['text.usetex'] = True

def read_multispec(input_file):
    """
    This function reads the input file and returns wavelength and flux.
    Recognizes IRAF multipspec spectra with different types of dispersion solution
    ES. 2020-10-21
    """
    try:
        hdu = fits.open(input_file)
    except Exception as e:
        print(f"Error while opening the input file: {e}")
    finally:
        header = hdu[0].header
        spectrum = hdu[0].data
        hdu.close()
    sizes = np.shape(spectrum)
    if len(sizes) == 1:
        nspec = 1
        npix = sizes[0]
    elif len(sizes) == 2:
        nspec = sizes[0]
        npix = sizes[1]
    elif len(sizes) >=3:
        nspec = sizes[-2]
        npix = sizes[-1]
        spectrum = spectrum[0]
    waves = np.zeros((nspec, npix), dtype=float)
    # try to recognize the type of dispersion
    if 'CTYPE1' in header:
        if header['CTYPE1'].strip() == 'LINEAR': # Linear dispersion solution
            crpix1 = header['CRPIX1']
            crval1 = header['CRVAL1']
            cd1_1 = header['CD1_1']
            wave = (np.arange(npix, dtype=float) + 1 - crpix1) * cd1_1 + crval1
            for i in range(nspec):
                waves[i, :] = wave
            if 'DC-FLAG' in header:
                if header['DC-FLAG'] == 1:
                    waves = 10**waves
        elif header['CTYPE1'].strip() == 'MULTISPE': # IRAF multispec data
            try:
                wat2 = header['WAT2_*']
            except Exception:
                print("Header does not contain keywords required for multispec data")
            finally:
                count_keys = len(wat2)
            long_wat2 = ""
            wave_params = np.zeros((nspec, 24), dtype=float)
            for i in wat2:
                key = header[i].replace('\'', '')
                if len(key) < 68: key += ' '
                long_wat2 += key
            for i in range(nspec):
                idx_b = long_wat2.find("\"", long_wat2.find("spec"+str(i+1)+" ="), -1)
                idx_e = long_wat2.find("\"", idx_b+1, -1)
                temparr = np.asarray(long_wat2[idx_b+1:idx_e].split())
                wave_params[i, 0:len(temparr)] = temparr
                if wave_params[i, 2] == 0 or wave_params[i, 2] == 1:
                    waves[i, :] = np.arange(npix, dtype=float) * wave_params[i, 4] \
                                + wave_params[i, 3]
                    if wave_params[i, 2] == 1:
                        waves[i, :] = 10**waves[i, :]
                else: # Non-linear solution. Not tested
                    waves[i, :] = nonlinearwave(npix, long_wat2[idx_b+1:idx_e])
        elif header['CTYPE1'].strip() == 'PIXEL':
            waves[:,:] = np.arange(npix)+1
    return waves,spectrum,header

def get_flux_std(filename, D, obsc, Texp, disp):
    wl, fl_tab = np.loadtxt(filename, unpack=True, dtype=float, comments='#')
    fl_phot = 5.03 * wl * fl_tab
    fl_obs = fl_phot * Texp * np.pi * D**2 / 4. * (1 - obsc**2) * disp
    return wl*10, fl_obs

def calibrate(wl_obs, fl_obs, wl_cal, fl_cal, hdr):
    nord = np.shape(wl_obs)[0]
    idx = np.argsort(wl_obs[:, np.shape(wl_obs)[1]//2+1])
    wl_dqe = wl_obs[:, np.shape(wl_obs)[1]//2+1][idx]
    fl_cal_rs = spectres.spectres(wl_dqe, wl_cal, fl_cal)
    fl_obs_rs = fl_obs[:, np.shape(wl_obs)[1]//2+1][idx]
    fig = plt.figure(figsize=(12,4), tight_layout=True)
    ax = fig.add_subplot(1,1,1)
    ax.plot(wl_dqe, fl_obs_rs/fl_cal_rs*100, 'ks', ms=5)
    ax.set_title(hdr['OBJNAME'].rstrip() + ', ' + hdr['DATE-OBS'])
    ax.set_xlabel(r"Wavelength, \AA", fontsize=fontsize)
    ax.set_ylabel(r"Throughput, \%", fontsize=fontsize)
    plt.show()
    return wl_dqe, fl_obs_rs/fl_cal_rs

if __name__ == "__main__":
    D = 240 # diameter in cm
    disp = 0.1166 # dispersion in Angstroems
    obsc = 0.3    # central obscuration of the telescope
    wl_obs, fl_obs, hdr = read_multispec(argv[2])
    Texp = float(hdr['EXPOSURE'])
    w_cal, fl_cal = get_flux_std(argv[1], D, obsc, Texp, disp)
    wc, fc = calibrate(wl_obs, fl_obs, w_cal, fl_cal, hdr)
    np.savetxt('dqe_'+(hdr['OBJNAME'].rstrip()).replace(' ', '')+hdr['DATE-OBS'].rstrip()+'.dat', \
              np.vstack((wc, fc)).transpose(), fmt="%.3f")

    exit(0)
