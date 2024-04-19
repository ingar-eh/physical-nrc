import matplotlib.pyplot as plt
import numpy as np


Ptx = np.linspace(0,300,1000)

r = 300
Prx = 10**((Ptx - 20*np.log10(r)) / 20) # calculate Prx and convert from decibel
SINR = Prx / (0.01*Prx + 1.8e-16)    

m = 2
N = 200
M = 3
Rc = 0.5
B = 3000
toh = 0.01
td = 0.25
PLR = 0.1
pc = 0.25
bb = 5 

BER = m * N * np.log2(M) * (0.2 * np.exp(-(3/(2 * (M - 1)) * SINR)))**(1 / Rc)

bBER = np.log(-(BER - PLR))
#bBER = -np.exp(bb*(BER - PLR))


f = np.log(m*(1 + pc)*N + B*(toh+td)) - np.log(m) - np.log(Rc) - np.log(B) - np.log(N)- np.log(np.log2(M)) \
- 1/bb * bBER

plt.figure()
plt.subplot(211)
plt.semilogy(Ptx, BER, color='tab:blue', label='BER')
plt.legend()

plt.subplot(212)
plt.plot(Ptx, f, color='tab:green', label='f')
#plt.plot(Ptx, SINR, color='tab:orange', label='SINR')
plt.legend()
plt.show()