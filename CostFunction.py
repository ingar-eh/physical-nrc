import autograd.numpy as np

class CostFunction:

    def __init__(self):
        c = 1500             # speed of sound in water
        t = 12               # temperature
        d = 2                # depth
        self.r = 300         # range
        self.td = self.r / c # transmission delay
        self.toh = 1e-3      # overhead delay
        self.L = 0.01        # energy spread threshold
        self.B = 3000        # bandwith in Hz
        self.pc = 0.25       # cyclic prefix ratio
        self.Rc = 0.5        # coding rate
        self.n_x = 0         # non-data subcarriers
        self.fk = (34+18)/2  # median frequency in kHz

        # barrier constants
        self.mmin = 1
        self.mmax = 40
        self.Nmin = 1
        self.Nmax = 2000
        self.Mmin = 2
        self.Mmax = 64  
        self.PLR = 0.001  # max loss ratio
        self.Pmin = 168.9 # min souself.rce level in dB re 1 µPa
        self.Pmax = 188.9 # max souself.rce level in dB re 1 µPa

        # absorbtion
        A1 = 1.03e-08 + 2.36e-10 * t - 5.22e-12 * t**2
        A2 = 5.62e-08 + 7.52e-10 * t
        A3 = (55.9 - 2.37 * t + 4.77e-02 * t**2 - 3.48e-04 * t**3) * 10**(-15)

        f1 = 1.32e+3 * (t + 273.1) * np.exp(-1700 / (t + 273.1))
        f2 = 1.55e+7 * (t + 273.1) * np.exp(-3052 / (t + 273.1))

        P1 = 1
        P2 = 1 - 10.3e-05 * d + 3.7e-09 * d**2
        P3 = 1 - 3.84e-05 * d + 7.57e-10 * d**2

        self.absorbtion = A1 * P1 * (f1 * self.fk**2) / (f1**2 + self.fk**2) + A2 * P2 * (f2 * self.fk**2) / (f2**2 + self.fk**2) + A3 * P3 * self.fk**2 # 10log10(a) Fisher Simmons

        # noise
        s = 0 # shipping factor. between 0 and 1
        w = 0 # wind speed in m/s
        
        Nt = 17 - 30 * np.log10(self.fk)                                               # turbulence
        Ns = 40 + 20 * (s-0.5) + 26 * np.log10(self.fk) - 60 * np.log10(self.fk+0.03)  # shipping
        Nw = 50 + 7.5 * np.sqrt(w) + 20 * np.log10(self.fk) - 40 * np.log(self.fk+0.4) # waves
        Nth = -15 + 20 * np.log10(self.fk)                                             # thermal
  
        noise = 10*np.log10(10**(Nt/10) + 10**(Ns/10) + 10**(Nw/10) + 10**(Nth/10)) + 10*np.log10(3000) - 170.8
        self.Pn = 10**(noise / 10)


    def get_fn(self, x: np.array, bb) -> np.array:

        # optimizable variables
        m = x[0]   # symbols per packet
        N = x[1]   # subcarriers
        M = x[2]   # modulation order
        Ptx = x[3] # transmit power

        # calculate pl
        Prx = 10**((Ptx - 20*np.log10(self.r) - self.absorbtion * self.r * 2 - 170.8) / 10)              # received power
        SINR = Prx / (self.L*Prx + self.Pn)                                                              # signal to interference plus noise ratio
        pl = m * (N - self.n_x) * np.log2(M) * (0.2 * np.exp(-(3/(2 * (M - 1)) * SINR)))**(1 / self.Rc)  # packet loss ratio


        # barrier functions / constraints
        bm = np.log(-(self.self.mmin - m)) + np.log(-(self.mmax - 40))      # 1 <  m  < 40
        bN = np.log(-(self.n_x + self.Nmin - N)) + np.log(-(N - self.Nmax)) # 1 <  N  < 2000
        bM = np.log(-(self.Mmin - M)) + np.log(-(M - self.Mmax))            # 2 <  M  < 64
        bpl = np.log(-(pl - self.PLR))                                      # pl < self.PLR
        bPtx = np.log(-(self.Pmin - Ptx)) + np.log(-(Ptx - self.Pmax))      # 168.9 < Ptx < 188.9

        # cost function
        f = np.log(m*(1 + self.pc)*N + self.B*(self.toh+self.td)) - np.log(m) - np.log(self.Rc) - np.log(self.B) - np.log(N - self.n_x) - np.log(np.log2(M)) \
                    - 1 / bb * (bm + bN + bM + bpl + bPtx)
        
        return f
    

    def get_gradient_fn(self, x: np.array, bb) -> np.array:

        # optimizable variables
        m = x[0]   # symbols per packet
        N = x[1]   # subcarriers
        M = x[2]   # modulation order
        Ptx = x[3] # transmit power

        # compacting expressions
        p = 1 + self.pc
        D = (self.toh + self.td) * self.B

        # gradient
        dJdm = N*p/(D + N*m*p) - bb * np.exp((self.self.mmin - m) * bb) + bb * np.exp((m - self.mmax) * bb)
        dJdN = m*p/(D + N*m*p) - 1/(N - self.n_x) - bb * np.exp((self.n_x + self.Nmin - N) * bb) + bb * np.exp((N - self.Nmax) * bb)
        dJdM = -1/(M*np.log(M)) - bb * np.exp((self.Mmin - M) * bb) + bb * np.exp((M - self.Mmax) * bb)
        dJdPtx = bb * np.exp((self.Pmin - Ptx) * bb) + bb * np.exp((Ptx - self.Pmax) * bb)
        
        # return np.array([dJdm, dJdN, dJdM])
        return np.array([dJdm, dJdN, dJdM, dJdPtx])
    

    def get_hessian_fn(self, x: np.array, bb) -> np.array:

        # optimizable variables
        m = x[0]   # symbols per packet
        N = x[1]   # subcarriers
        M = x[2]   # modulation order
        Ptx = x[3] # transmit power

        # compacting expressions
        p = 1 + self.pc
        D = (self.toh + self.td) * self.B
        DD = (D + N * m * p)

        # hessian
        d2Jdm2 = -(N * p)**2 / DD**2 + m**(-2) + (bb**2) * np.exp((self.mmin - m) * bb) + (bb**2) * np.exp((m - self.mmax) * bb)
        d2JdmdN = -(N * M * p**2) / DD**2 + p / DD
        d2JdN2 = -(m * p)**2 / DD**2 + (N - self.n_x)**(-2) + (bb**2) * np.exp((self.n_x + self.Nmin - N) * bb) + (bb**2) * np.exp((N - self.Nmax) * bb)
        d2JdM2 = 1 / (M**2 * np.log(M)) + 1 / (M * np.log(M))**2 + (bb**2) * np.exp((self.Mmin - M) * bb) + (bb**2) * np.exp((M - self.Mmax) * bb)
        d2JdPtx2 = (bb**2) * np.exp((self.Pmin - Ptx) * bb) + (bb**2) * np.exp((Ptx - self.Pmax) * bb)
        
        # return np.array([[ d2Jdm2, d2JdmdN,      0],
        #                  [d2JdmdN,  d2JdN2,      0],
        #                  [      0,       0, d2JdM2]])
        return   np.array([[ d2Jdm2, d2JdmdN,      0,        0],
                           [d2JdmdN,  d2JdN2,      0,        0],
                           [      0,       0, d2JdM2,        0],
                           [      0,       0,      0, d2JdPtx2]])            
        
