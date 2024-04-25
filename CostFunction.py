import autograd.numpy as np

class CostFunction:

    @staticmethod
    def get_fn(x: np.array, bb) -> np.array:

        # optimizable variables
        m = x[0]   # symbols per packet
        N = x[1]   # subcarriers
        M = x[2]   # modulation order
        P = x[3]   # transmit power; watts

        c = 1500        # speed of sound in water; m/s
        t = 12          # temperature; celcius
        d = 2           # depth; meters
        r = 300         # range; meters
        td = r / c      # transmission delay; seconds
        toh = 1e-3      # overhead delay; seconds
        L = 0.01        # energy spread threshold
        B = 3000        # bandwith; Hz
        pc = 0.25       # cyclic prefix ratio
        Rc = 0.5        # coding rate
        n_x = 0         # non-data subcarriers
        fk = (34+18)/2  # median frequency; kHz

        # barrier constants
        mmin = 1
        mmax = 40
        Nmin = 1
        Nmax = 2000
        Mmin = 2
        Mmax = 64  
        PLR = 0.001 # max loss ratio
        Pmin = 5.5  # min power; watts
        Pmax = 65   # max power; watts

        # absorbtion
        A1 = 1.03e-08 + 2.36e-10 * t - 5.22e-12 * t**2
        A2 = 5.62e-08 + 7.52e-10 * t
        A3 = (55.9 - 2.37 * t + 4.77e-02 * t**2 - 3.48e-04 * t**3) * 10**(-15)

        f1 = 1.32e+3 * (t + 273.1) * np.exp(-1700 / (t + 273.1))
        f2 = 1.55e+7 * (t + 273.1) * np.exp(-3052 / (t + 273.1))

        P1 = 1
        P2 = 1 - 10.3e-05 * d + 3.7e-09 * d**2
        P3 = 1 - 3.84e-05 * d + 7.57e-10 * d**2

        absorbtion = A1 * P1 * (f1 * fk**2) / (f1**2 + fk**2) + A2 * P2 * (f2 * fk**2) / (f2**2 + fk**2) + A3 * P3 * fk**2 # dB re 1µPa/km

        # noise
        s = 0 # shipping factor; between 0 and 1
        w = 0 # wind speed; m/s
        
        Nt = 17 - 30 * np.log10(fk)                                            # turbulence
        Ns = 40 + 20 * (s-0.5) + 26 * np.log10(fk) - 60 * np.log10(fk+0.03)    # shipping
        Nw = 50 + 7.5 * np.sqrt(w) + 20 * np.log10(fk) - 40 * np.log10(fk+0.4) # waves
        Nth = -15 + 20 * np.log10(fk)                                          # thermal
  
        noise = 20*np.log10(10**(Nt/10) + 10**(Ns/10) + 10**(Nw/10) + 10**(Nth/10)) + 20*np.log10(B) - 170.8
        Pn = 10**(noise / 10)

        # calculate pl
        Ptx = 10*np.log10(P) + 170.8                                                          # transmitted power; dB watts
        Prx = 10**((Ptx - 20*np.log10(r) - absorbtion * r/1000 * 2 - 170.8) / 10)             # received power; watts
        SINR = Prx / (L*Prx + Pn)                                                             # signal to interference plus noise ratio
        pl = m * (N - n_x) * np.log2(M) * (0.2 * np.exp(-(3/(2 * (M - 1)) * SINR)))**(1 / Rc) # packet loss ratio


        # barrier functions / constraints
        bm = np.log(-(mmin - m)) + np.log(-(m - mmax))
        bN = np.log(-(n_x + Nmin - N)) + np.log(-(N - Nmax))
        bM = np.log(-(Mmin - M)) + np.log(-(M - Mmax))
        bpl = np.log(-(pl - PLR))
        bPtx = np.log(-(Pmin - Ptx)) + np.log(-(Ptx - Pmax))

        # cost function
        f = np.log(m*(1 + pc)*N + B*(toh+td)) - np.log(m) - np.log(Rc) - np.log(B) - np.log(N - n_x) - np.log(np.log2(M)) - np.log(1 - pl) \
                    - 1 / bb * (bm + bN + bM + bpl + bPtx)
        
        return f