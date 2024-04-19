import numpy
import random

import numpy as np
import autograd.numpy as np
from autograd import jacobian

import math

class SimulationFunctionXTX_BTX:
    """This class representing the simulation function. This class can be used to calculate the gradient and hessian of the mentioned function. """
    @staticmethod
    def get_fn(x: np.array, bb, pc=0.25, sea_cond=np.array([12, 40, 50, 3]), B=3000, toh=1e-3, n_x=0,
               Rc=0.5) -> np.array:
        """This method can be used to calculate the outcome of the function for each given Xi and Bi. X = [n, N, M, pc]"""
        m = x[0]
        N = x[1]
        M = x[2]
        Ptx = x[3]

        # constants
        
        c = 1500 # speed of sound in water
        t = sea_cond[0] # temperature
        d = sea_cond[2] # depth
        r = sea_cond[3] # range
        PLR = 0.001 # package loss ratio
        fk = (34+18)/2 # median frequency in kHz
        td = r / c # transmission delay
        L = 0.01  # energy spread threshold. assumed value

        # absorbtion

        A1 = 1.03e-08 + 2.36e-10 * t - 5.22e-12 * t**2
        A2 = 5.62e-08 + 7.52e-10 * t
        A3 = (55.9 - 2.37 * t + 4.77e-02 * t**2 - 3.48e-04 * t**3) * 10**(-15)

        f1 = 1.32e+3 * (t + 273.1) * np.exp(-1700 / (t + 273.1))
        f2 = 1.55e+7 * (t + 273.1) * np.exp(-3052 / (t + 273.1))

        P1 = 1
        P2 = 1 - 10.3e-05 * d + 3.7e-09 * d**2
        P3 = 1 - 3.84e-05 * d + 7.57e-10 * d**2

        #absorbtion = (0.1 * fk**2) / (1 + fk**2) + (40 * fk**2) / (4100 + fk**2) + 2.75e-04 * fk**2 + 0.003 # dB/km for f in kHz ## only good for temperatures 4C and depth of 1000m
        absorbtion = A1 * P1 * (f1 * fk**2) / (f1**2 + fk**2) + A2 * P2 * (f2 * fk**2) / (f2**2 + fk**2) + A3 * P3 * fk**2 # 10log10(a) Fisher Simmons

        # noise

        s = 0 # shipping factor; 0.5 for moderate shipping
        w = 0 # wind speed in m/s
        
        Nt = 17 - 30 * np.log10(fk) # turbulence
        Ns = 40 + 20 * (s-0.5) + 26 * np.log10(fk) - 60 * np.log10(fk+0.03) # shipping
        Nw = 50 + 7.5 * np.sqrt(w) + 20 * np.log10(fk) - 40 * np.log(fk+0.4) # waves
        Nth = -15 + 20 * np.log10(fk) # thermal

        #q = 1 / (2**M - 1) # quantization
        #qn = q**2 / 12  # quantization noise
        
        noise = 10*np.log10(10**(Nt/10) + 10**(Ns/10) + 10**(Nw/10) + 10**(Nth/10)) - 170.8 # 10log(f)
        Pn = 10**(noise / 10)
        #Pn = 10**(50/20)

        Prx = 10**((Ptx - 20*np.log10(r) - absorbtion * r * 2 - 170.8) / 10) # calculate Prx and convert from decibel
        SINR = Prx / (L*Prx + Pn)
        BER = m * (N - n_x) * np.log2(M) * (0.2 * np.exp(-(3/(2 * (M - 1)) * SINR)))**(1 / Rc)
        #SINR = 100

        # barrier
    
        bm = np.log(-(1 - m)) + np.log(-(m - 40)) # 1 < m < 40
        bN = np.log(-(n_x + 1 - N)) + np.log(-(N - 2000)) # 400 < N < 2000
        bM = np.log(-(2 - M)) + np.log(-(M - 64)) # 2 < M < 64
        bBER = np.log(-(BER - PLR)) # BER < PLR
        bPtx = np.log(-(168.9 - Ptx)) + np.log(-(Ptx - 188.9))

        # cost function

        f = np.log(m*(1 + pc)*N + B*(toh+td)) - np.log(m) - np.log(Rc) - np.log(B) - np.log(N - n_x) - np.log(np.log2(M)) \
                    - 1 / bb * (bm + bN + bM + bBER + bPtx) # constraints

        try:
            if math.isnan(float(f)):
                pass
        except:
            pass
        
        print(f"SINR:{SINR}\n")
        print(f"BER:{BER}\n")
        print(f"f:{f}")
        return f
    
    @staticmethod
    def get_gradient_fn(x: np.array, bb, pc=0.25, sea_cond=np.array([12, 40, 20, 3]), toh=1e-3, B=3000, eta=1, Rc=0.5,
                        pb=0.1) -> np.array:
        """This method can be used to calculate the gradient for any given Xi."""
        m = x[0]
        N = x[1]
        M = x[2]
        Ptx = x[3]

        p = 1 + pc
        c = 1500
        n_x = 0
        r = sea_cond[3]
        td = r / c
        D = (toh + td) * B

        # new gradient

        dm = N*p/(D + N*m*p) - bb * np.exp((1 - m) * bb) + bb * np.exp((m - 40) * bb)
        dN = m*p/(D + N*m*p) - 1/(N - n_x) - bb * np.exp((n_x + 1 - N) * bb) + bb * np.exp((N - 2000) * bb)
        dM = -1/(M*np.log(M)) - bb * np.exp((2 - M) * bb) + bb * np.exp((M - 64) * bb)
        dPtx = bb * np.exp((168.9 - Ptx) * bb) + bb * np.exp((Ptx - 188.9) * bb)
        
        answa0 = np.array([dm, dN, dM])
        answa1 = np.array([dm, dN, dM, dPtx])
        return answa1
    
    @staticmethod
    def get_hessian_fn(x: np.array, bb, pc=0.25, *, sea_cond=np.array([12, 40, 20, 3]), toh=1e-3, B=3000, Rc=0.5,
                       eta=1,
                       pb=1) -> np.array:
        """This method can be used to calculate the hessian for any given Xi."""
        m = x[0]
        N = x[1]
        M = x[2]
        Ptx = x[3]

        
        c = 1500
        n_x = 0
        r = sea_cond[3]
        td = r / c
        D = (toh + td) * B
        p = 1 + pc
        DD = (D + N * m * p)

        # new hessian

        dmm = -(N * p)**2 / DD**2 + m**(-2) + (bb**2) * np.exp((1 - m) * bb) + (bb**2) * np.exp((m - 40) * bb)
        dmN = -(N * M * p**2) / DD**2 + p / DD
        dNN = -(m * p)**2 / DD**2 + (N - n_x)**(-2) + (bb**2) * np.exp((n_x + 1 - N) * bb) + (bb**2) * np.exp((N - 2000) * bb)
        dMM = 1 / (M**2 * np.log(M)) + 1 / (M * np.log(M))**2 + (bb**2) * np.exp((2 - M) * bb) + (bb**2) * np.exp((M - 64) * bb)
        dPtxPtx = (bb**2) * np.exp((168.9 - Ptx) * bb) + (bb**2) * np.exp((Ptx - 188.9) * bb)
        
        answa0 = np.array([[dmm,   dmN,   0],
                           [dmN,   dNN,   0],
                           [  0,     0, dMM]])
        answa1 = np.array([[dmm,   dmN,   0,       0],
                           [dmN,   dNN,   0,       0],
                           [  0,     0, dMM,       0],
                           [  0,     0,   0, dPtxPtx]])            
        
        return answa1
