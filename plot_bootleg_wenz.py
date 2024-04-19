import matplotlib.pyplot as plt
import numpy as np

fk = np.linspace(1e-3,1e+3,1000000)

s = 0.5 # shipping factor; 0.5 for moderate shipping
w = 5 # wind speed in m/s

Nt = 17 - 30 * np.log10(fk) # turbulence
Ns = 40 + 20 * (s-0.5) + 26 * np.log10(fk) - 60 * np.log10(fk+0.03) # shipping
Nw = 50 + 7.5 * np.sqrt(w) + 20 * np.log10(fk) - 40 * np.log(fk+0.4) # waves
Nth = -15 + 20 * np.log10(fk) # thermal


fig, ax = plt.subplots()
#ax.set_xlim([xmin,xmax])
ax.set_ylim([0,110])
ax.semilogx(fk, Nt, color='tab:green', label='turbulence')
ax.semilogx(fk, Ns, color='tab:blue', label='shipping')
ax.semilogx(fk, Nw, color='tab:red', label='wind')
ax.semilogx(fk, Nth, color='tab:purple', label='thermal')
ax.legend()
plt.show()
