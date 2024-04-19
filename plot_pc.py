import matplotlib.pyplot as plt
import numpy as np


#Ptx = np.linspace(120,300,1000)
Ptx = 200


Prx = 10**((Ptx - 20*np.log10(300)) / 20) # calculate Prx and convert from decibel
SINR = Prx / (0.01*Prx + 21)    

m = 6
N = 800
M = 22
Rc = 0.5
B = 3000
toh = 0.01
td = 0.25
PLR = 0.1
pc = 0.25
bb = 5

pc = np.linspace(0.001,0.999,1000)

bpc = np.log(-(0 - pc)) + np.log(-(pc - 1))


f0 = np.log(m*(1 + pc)*N + B*(toh+td)) - np.log(m) - np.log(Rc) - np.log(B) - np.log(N)- np.log(np.log2(M)) - 1/5 * bpc
f1 = np.log(m*(1 + pc)*N + B*(toh+td)) - np.log(m) - np.log(Rc) - np.log(B) - np.log(N)- np.log(np.log2(M)) - 1/15 * bpc
f2 = np.log(m*(1 + pc)*N + B*(toh+td)) - np.log(m) - np.log(Rc) - np.log(B) - np.log(N)- np.log(np.log2(M)) - 1/150 * bpc


fig, ax = plt.subplots()
#ax.set_xlim([xmin,xmax])
ax.set_ylim([-8.75,-7.25])
ax.plot(pc, f0, color='tab:green', label='t=5')
ax.plot(pc, f1, color='tab:blue', label='t=15')
ax.plot(pc, f2, color='tab:red', label='t=150')
ax.legend()
plt.show()