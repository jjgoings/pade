import numpy as np
from pade import pade
import matplotlib.pyplot as plt

w = 2.0
t = np.arange(0.0, 20.0, 0.01)
signal = np.sin(w*t) + np.sin(2*w*t) + np.sin(4*w*t) 

plt.plot(t,signal)
plt.savefig('signal.png')

plt.clf()
plt.cla()


fw, frequency = pade(t,signal,w_max=10.0)

plt.plot(frequency,np.imag(fw))
plt.savefig('fsignal.png')
plt.show()


