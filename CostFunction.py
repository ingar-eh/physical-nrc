import numpy as np

class CostFunction:

    @staticmethod
    def get_fn(x: np.array, bb) -> np.array:

        # optimizable variables

        m = x[0]   # symbols per packet
        N = x[1]   # subcarriers
        M = x[2]   # modulation order
        Ptx = x[3] # transmit power

        # constants
        
        c = 1500       # speed of sound in water
        t = 12         # temperature
        d = 2          # depth
        r = 300        # range
        td = r / c     # transmission delay
        toh = 1e-3     # overhead delay
        L = 0.01       # energy spread threshold
        B = 3000       # bandwith in Hz
        pc = 0.25      # cyclic prefix ratio
        Rc = 0.5       # coding rate
        n_x = 0        # non-data subcarriers
        fk = (34+18)/2 # median frequency in kHz

        # absorbtion

        A1 = 1.03e-08 + 2.36e-10 * t - 5.22e-12 * t**2
        A2 = 5.62e-08 + 7.52e-10 * t
        A3 = (55.9 - 2.37 * t + 4.77e-02 * t**2 - 3.48e-04 * t**3) * 10**(-15)

        f1 = 1.32e+3 * (t + 273.1) * np.exp(-1700 / (t + 273.1))
        f2 = 1.55e+7 * (t + 273.1) * np.exp(-3052 / (t + 273.1))

        P1 = 1
        P2 = 1 - 10.3e-05 * d + 3.7e-09 * d**2
        P3 = 1 - 3.84e-05 * d + 7.57e-10 * d**2

        absorbtion = A1 * P1 * (f1 * fk**2) / (f1**2 + fk**2) + A2 * P2 * (f2 * fk**2) / (f2**2 + fk**2) + A3 * P3 * fk**2 # 10log10(a) Fisher Simmons

        # noise

        s = 0 # shipping factor; 0.5 for moderate shipping
        w = 0 # wind speed in m/s
        
        Nt = 17 - 30 * np.log10(fk)                                          # turbulence
        Ns = 40 + 20 * (s-0.5) + 26 * np.log10(fk) - 60 * np.log10(fk+0.03)  # shipping
        Nw = 50 + 7.5 * np.sqrt(w) + 20 * np.log10(fk) - 40 * np.log(fk+0.4) # waves
        Nth = -15 + 20 * np.log10(fk)                                        # thermal

        #q = 1 / (2**M - 1) # quantization
        #qn = q**2 / 12  # quantization noise
        
        noise = 10*np.log10(10**(Nt/10) + 10**(Ns/10) + 10**(Nw/10) + 10**(Nth/10)) + 10*np.log10(3000) - 170.8
        Pn = 10**(noise / 10) # convert from decibel to power

        Prx = 10**((Ptx - 20*np.log10(r) - absorbtion * r * 2 - 170.8) / 10) # calculate Prx and convert from decibel to power
        SINR = Prx / (L*Prx + Pn)
        BER = m * (N - n_x) * np.log2(M) * (0.2 * np.exp(-(3/(2 * (M - 1)) * SINR)))**(1 / Rc)

        # barrier function and constants

        mmin = 1
        mmax = 40
        Nmin = 1
        Nmax = 2000
        Mmin = 2
        Mmax = 64  
        PLR = 0.001  # max loss ratio
        Pmin = 168.9 # min source level in dB re 1 µPa
        Pmax = 188.9 # max source level in dB re 1 µPa
    
        bm = np.log(-(mmin - m)) + np.log(-(mmax - 40))      #     1 <  m  < 40
        bN = np.log(-(n_x + Nmin - N)) + np.log(-(N - Nmax)) #     1 <  N  < 2000
        bM = np.log(-(Mmin - M)) + np.log(-(M - Mmax))       #     2 <  M  < 64
        bBER = np.log(-(BER - PLR))                          #         BER < PLR
        bPtx = np.log(-(Pmin - Ptx)) + np.log(-(Ptx - Pmax)) # 168.9 < Ptx < 188.9

        # cost function

        f = np.log(m*(1 + pc)*N + B*(toh+td)) - np.log(m) - np.log(Rc) - np.log(B) - np.log(N - n_x) - np.log(np.log2(M)) \
                    - 1 / bb * (bm + bN + bM + bBER + bPtx) # constraints
        
        return f
    
    @staticmethod
    def get_gradient_fn(x: np.array, bb) -> np.array:

        # optimizable variables

        m = x[0]   # symbols per packet
        N = x[1]   # subcarriers
        M = x[2]   # modulation order
        Ptx = x[3] # transmit power

        # constants
        
        c = 1500       # speed of sound in water
        r = 300        # range
        td = r / c     # transmission delay
        toh = 1e-3     # overhead delay
        B = 3000       # bandwith in Hz
        pc = 0.25      # cyclic prefix ratio
        n_x = 0        # non-data subcarriers

        # barrier constants

        mmin = 1
        mmax = 40
        Nmin = 1
        Nmax = 2000
        Mmin = 2
        Mmax = 64  
        PLR = 0.001  # max loss ratio
        Pmin = 168.9 # min source level in dB re 1 µPa
        Pmax = 188.9 # max source level in dB re 1 µPa
        Pmin = 168.9   # min source level in dB re 1 µPa
        Pmax = 188.9   # max source level in dB re 1 µPa

        # compacting expressions

        p = 1 + pc
        D = (toh + td) * B

        # gradient

        dJdm = N*p/(D + N*m*p) - bb * np.exp((mmin - m) * bb) + bb * np.exp((m - mmax) * bb)
        dJdN = m*p/(D + N*m*p) - 1/(N - n_x) - bb * np.exp((n_x + Nmin - N) * bb) + bb * np.exp((N - Nmax) * bb)
        dJdM = -1/(M*np.log(M)) - bb * np.exp((Mmin - M) * bb) + bb * np.exp((M - Mmax) * bb)
        dJdPtx = bb * np.exp((Pmin - Ptx) * bb) + bb * np.exp((Ptx - Pmax) * bb)
        
        # return np.array([dJdm, dJdN, dJdM])
        return np.array([dJdm, dJdN, dJdM, dJdPtx])
    
    @staticmethod
    def get_hessian_fn(x: np.array, bb) -> np.array:

        # optimizable variables

        m = x[0]   # symbols per packet
        N = x[1]   # subcarriers
        M = x[2]   # modulation order
        Ptx = x[3] # transmit power

        # constants
        
        c = 1500       # speed of sound in water
        r = 300        # range
        td = r / c     # transmission delay
        toh = 1e-3     # overhead delay
        B = 3000       # bandwith in Hz
        pc = 0.25      # cyclic prefix ratio
        n_x = 0        # non-data subcarriers

        # barrier constants

        mmin = 1
        mmax = 40
        Nmin = 1
        Nmax = 2000
        Mmin = 2
        Mmax = 64  
        PLR = 0.001  # max loss ratio
        Pmin = 168.9 # min source level in dB re 1 µPa
        Pmax = 188.9 # max source level in dB re 1 µPa
        Pmin = 168.9   # min source level in dB re 1 µPa
        Pmax = 188.9   # max source level in dB re 1 µPa
        
        # compacting expressions

        p = 1 + pc
        D = (toh + td) * B
        DD = (D + N * m * p)

        # new hessian

        d2Jdm2 = -(N * p)**2 / DD**2 + m**(-2) + (bb**2) * np.exp((1 - m) * bb) + (bb**2) * np.exp((m - 40) * bb)
        d2JdmdN = -(N * M * p**2) / DD**2 + p / DD
        d2JdN2 = -(m * p)**2 / DD**2 + (N - n_x)**(-2) + (bb**2) * np.exp((n_x + 1 - N) * bb) + (bb**2) * np.exp((N - 2000) * bb)
        d2JdM2 = 1 / (M**2 * np.log(M)) + 1 / (M * np.log(M))**2 + (bb**2) * np.exp((2 - M) * bb) + (bb**2) * np.exp((M - 64) * bb)
        d2JdPtx2 = (bb**2) * np.exp((Pmin - Ptx) * bb) + (bb**2) * np.exp((Ptx - Pmax) * bb)
        
        # return np.array([[ d2Jdm2, d2JdmdN,      0],
        #                  [d2JdmdN,  d2JdN2,      0],
        #                  [      0,       0, d2JdM2]])
        return   np.array([[ d2Jdm2, d2JdmdN,      0,        0],
                           [d2JdmdN,  d2JdN2,      0,        0],
                           [      0,       0, d2JdM2,        0],
                           [      0,       0,      0, d2JdPtx2]])            
        
