#!/usr/bin/env python3

from sys import exit
import argparse
import numpy as np
from scipy.interpolate import interp1d, UnivariateSpline
import matplotlib.pyplot as plt
import matplotlib as mpl

fontsize = 12
mpl.rcParams['xtick.labelsize'] = fontsize
mpl.rcParams['ytick.labelsize'] = fontsize
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'cm'
mpl.rcParams['text.usetex'] = True

D = 240 # diameter in cm
disp = 0.1166 # dispersion in Angstroems
Apix = 3.1415 #1.116   # Area of sky integrated in one dispersion element: 0.6 * (3.1 * 0.6)
               # where 0.6"/pix is a plate scale, 3.1 is a height of an order in pix
obsc = 0.3    # central obscuration of the telescope

gain = 0.55 # CCD gain
rdnoise = 2.0 # CCD readout noise

teff = {'O5': 42000., 'O9': 34000., 'B0': 30000., 'B2': 20900., 'B5': 15200., \
        'B8': 11400., 'A0': 9800., 'A2': 9000., 'A5': 8200., 'F0': 7300., \
        'F2': 7000., 'F5': 6700., 'F8': 6300., 'G0': 5900., 'G2': 5800., \
        'G5': 5500., 'G8': 5300., 'K0': 5100., 'K2': 4800., 'K5': 4400., \
        'M0': 3800., 'M2': 3500., 'M5': 3200., 'A8': 7900} # Effective temperature (approx.)

bc = {'O5': -4.40, 'O9': -3.33, 'B0': -3.16, 'B2': -2.35, 'B5': -1.46, \
      'B8': -0.80, 'A0': -0.30, 'A2': -0.20, 'A5': -0.15, 'F0': -0.09, \
      'F2': -0.11, 'F5': -0.14, 'F8': -0.16, 'G0': -0.18, 'G2': -0.20, \
      'G5': -0.21, 'G8': -0.40, 'K0': -0.31, 'K2': -0.42, 'K5': -0.72, \
      'M0': -1.38, 'M2': -1.89, 'M5': -2.73, 'A8': -0.12} # Bolometric correction

wl_dqe = np.array([4099, 4147, 4196, 4246, 4297, 4349, 4403, 4458, 4514, 4572, \
         4631, 4692, 4755, 4819, 4885, 4953, 5023, 5095, 5168, 5244, 5323, \
         5403, 5487, 5572, 5661, 5752, 5846, 5944, 6045, 6149, 6257, 6368, \
         6484, 6604, 6729, 6858, 6993], dtype=float) # Reference wavelength for DQE

dqe = np.array([0.00040279, 0.00069506, 0.00020847, 0.00047916, 0.00263364, 0.00351451, \
      0.00756952, 0.00963485, 0.01164158, 0.01287583, 0.01624699, 0.01841101, \
      0.0208349, 0.02585675, 0.02283765, 0.02547978, 0.02682312, 0.02927178, \
      0.0270037, 0.03192889, 0.03143859, 0.03107558, 0.03221363, 0.03090602, \
      0.02931054, 0.02654877, 0.02468006, 0.02178863, 0.02011702, 0.01861112, \
      0.01943067, 0.02165528, 0.0280808, 0.03276005, 0.03875882, 0.04372878, \
      0.04754378]) # Measured DQE of MRES

Cmoon = {'new': [22.0, 22.7, 21.8, 20.9, 19.9],  # '3': [21.5, 22.4, 21.7, 20.8, 19.9], \
         'quarter': [19.9, 21.6, 21.4, 20.6, 19.7], # 10: [18.5, 20.7, 20.7, 20.3, 19.5], \
         'full': [17.0, 19.5, 20.0, 19.9, 19.2]} # Sky brightness above Cerro Tololo (NOAO Newletter #10)
         # For TNO these values are obviously too optimistic because of light pollution and different scattering

wl_ubvri = np.array([3650., 4450., 5510., 6580., 8060.])

calibrator = {'O5': 'fluxes/hr3165_sc_fl.dat', 'O9': 'fluxes/hr1899_sc_fl.dat', \
              'B0': 'fluxes/hr6165_sc_fl.dat', 'B2': 'fluxes/hr4679_sc_fl.dat', \
              'B5': 'fluxes/hr4773_sc_fl.dat', 'B8': 'fluxes/hd68903_sc_fl.dat', \
              'A0': 'fluxes/hr7001_sc_fl.dat', 'A2': 'fluxes/hd162772_sc_fl.dat', \
              'A5': 'fluxes/hd153015_sc_fl.dat', 'F0': 'fluxes/hd19521_sc_fl.dat', \
              'F2': 'fluxes/hd20127_sc_fl.dat', 'F5': 'fluxes/hd199999_sc_fl.dat', \
              'F8': 'fluxes/hd171888_sc_fl.dat', 'G0': 'fluxes/hd83616_sc_fl.dat', \
              'G2': 'fluxes/hr5459_sc_fl.dat', 'G5': 'fluxes/hr6623_sc_fl.dat', \
              'G8': 'fluxes/hr7602_sc_fl.dat', 'K0': 'fluxes/hr7462_sc_fl.dat', \
              'K2': 'fluxes/hr1084_sc_fl.dat', 'K5': 'fluxes/hr6705_sc_fl.dat', \
              'M0': 'fluxes/hr7635_sc_fl.dat', 'M2': 'fluxes/hr1162_sc_fl.dat', \
              'M5': 'fluxes/hr8621_sc_fl.dat', 'A8': 'fluxes/hr1444_sc_fl.dat'}
              # Scaled to V=0 fluxes of standard stars from Biryukov et al. (1998) and Alekseeva et al. (1997)

k = np.array([0.42, 0.23, 0.12, 0.09, 0.06]) # Extinction at TNO (UBVRI, data by Ram Kesh)

def E0(lam, Teff, BC):
    return (8.48 * 10**(34) * 10**(-0.4 * BC)) / (Teff**4 * lam**4 * (np.exp((1.44 * 10**8)/(Teff * lam)) - 1))

def interpolate_flux(calib_w, w0, f0):
    interpolator = interp1d(w0, f0, fill_value="extrapolate")
    return interpolator(calib_w)

def get_calibration(db, sptype, wl_ref):
    try:
        wl, fl = np.loadtxt(db[sptype], usecols=(0,1), unpack=True)
    except Exception as e:
        print(e)
    else:
        fl0 = interpolate_flux(wl_ref, wl, fl)
        return fl0

def get_sky_mag(db, phase, wl_ref):
    Vsky = np.asarray(db[phase]) - 2.5*np.log10(Apix)
    coef = np.polyfit(wl_ubvri, Vsky, 4) # Fit polynomial of 4th order
    fl_sun = get_calibration(calibrator, 'G2', wl_ref)
    return  fl_sun * 10**(-0.4*np.polyval(coef, wl_ref))

def snr_prep(wl_ref, Vmag, sptype, Moon):
    dqe_cont = interpolate_flux(wl_ref, wl_dqe, dqe)
    dqe_cont[np.where(dqe_cont < 0.)] = 0.0
    fl0 = get_calibration(calibrator, sptype, wl_ref) * gain
    Bsky = get_sky_mag(Cmoon, Moon, wl_ref) * gain
    Ncont = fl0 * np.pi * D**2 / 4. * (1 - obsc**2) * disp * dqe_cont * 10**(-0.4 * Vmag)
    Ncont[np.where(Ncont <= 0)] = 1.
    A = Ncont + (3.1 * Bsky * np.pi * D**2 / 4. * (1 - obsc**2) * disp * dqe_cont)
    B = 3.1 * rdnoise**2
    return Ncont, A, B

def comp_snr(time_exp, w0, Vmag, sptype, Moon):
    wl_ref = np.arange(4000., 7025., 25.) # Reference wavelengths
    counts, a, b = snr_prep(wl_ref, Vmag, sptype, Moon)
    SNR_cont = (counts * time_exp) / (np.sqrt(time_exp * a + b))
    inter = interp1d(wl_ref, SNR_cont)
    return int(inter(w0)), wl_ref, np.ceil(SNR_cont)

def comp_time(SNR_req, w0, Vmag, sptype, Moon):
    wl_ref = np.arange(4000., 7025., 25.) # Reference wavelengths
    counts, a, b = snr_prep(wl_ref, Vmag, sptype, Moon)
    time_cont = 0.5 * SNR_req**2 * (a + np.sqrt(a**2 + 4*counts**2 * b / SNR_req**2)) / counts**2
    inter = interp1d(wl_ref, time_cont)
    return int(inter(w0)), wl_ref, np.ceil(time_cont)

def plot_graph(x, y, label, par, sptype, vmag, moon):
    fig = plt.figure(figsize=(9, 3), tight_layout=True)
    ax = fig.add_subplot(1,1,1)
    ax.plot(x, y, 'ks', ms=2)
    if label == "Expected time, s":
        ax.set_title("SNR %s, %s star, Vmag = %.1f, $s Moon" %(par, sptype, vmag, moon))
    else:
        ax.set_title("Texp = %s s, %s star, Vmag = %.1f, %s Moon" %(par, sptype, vmag, moon))
    ax.set_xlabel(r"Wavelength, \AA", fontsize=fontsize)
    ax.set_ylabel(label, fontsize=fontsize)
    plt.show()
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sptype', type=str, help="Spectral type among available [O5, O9, B0, B2, B5, B8, A0, A2, A5, A8, F0, F2, F5, F8, G0, G2, G5, G8, K0, K2, K5, M0, M2, M5]", \
                        default='A0')
    parser.add_argument('--moon', type=str, help="Lunar phase in days from new Moon. Available values are ['new', 'quarter', 'full']", \
                        default='14')
    parser.add_argument('--mag', type=float, help="Magnitude in V band", default=10.)
    parser.add_argument('--wave', type=float, help="Reference wavelength in Agstroems", default=5500.)
    parser.add_argument('--snr', help="Compute time, required for specified SNR", type=float, default=-99)
    parser.add_argument('--texp', help="Compute SNR from specified time", type=float, default=-99)
    parser.add_argument('--showall', help="Plot the diagram for the full range of wavelengths", action="store_true")
    args = parser.parse_args()

    if args.wave <= wl_dqe[0] or args.wave >= wl_dqe[-1]:
        print("Error: required wavelength is out of range %.0f-%.0f" %(wl_dqe[0], wl_dqe[-1]))
        exit(1)
    if args.snr == -99 and args.texp == -99:
        print("Error: Unknown operation. Please read help")
        exit(1)
    if args.texp != 99 and args.texp > 0:
        snr, wl, snr_wl = comp_snr(args.texp, args.wave, args.mag, args.sptype, args.moon)
        print("Expected SNR(%.0f) = %d in %d s" %(args.wave, int(snr), int(args.texp)))
        if args.showall:
            plot_graph(wl, snr_wl, "Expected SNR", int(args.texp), args.sptype, args.mag, args.moon)
    elif args.snr != 99 and args.snr > 0:
        texp, wl, texp_wl = comp_time(args.snr, args.wave, args.mag, args.sptype, args.moon)
        print("Expected Texp to achieve SNR(%.0f) = %d is %d s" %(args.wave, int(args.snr), int(texp)))
        if args.showall:
            plot_graph(wl, texp_wl, "Expected time, s", int(args.snr), args.sptype, args.mag, args.moon)

    exit(0)
