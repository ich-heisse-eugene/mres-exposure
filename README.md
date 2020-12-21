# mres-exposure
Exposure calculator for MRES

Allowed spectral types: O5, O9, B0, B2, B5, B8, A0, A2, A5, A8, F0, F2, F5, F8, G0, G2, G5, G8, K0, K2, K5, M0, M2, M5

Lunar phases: new, quarter, full

Examples of usage:

1. Compute the estimated SNR for an A5 star of 8.2 V mag. achieved during a single 600-seconds long exposure at 5490 A in bright night and display results for other wavelengths:

python3 mres_expose.py --sptype A5 --moon 'full' --mag 8.2 --wave 5490 --texp 600 --showall

2. Compute time, required for specified SNR:

python3 mres_expose.py --sptype A5 --moon 'full' --mag 8.2 --wave 5490 --snr 100

3. Display a short description of available options:

python3 mres_expose.py --help
