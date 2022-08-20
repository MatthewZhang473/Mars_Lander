
import numpy as np
import matplotlib.pyplot as plt
results = np.loadtxt('control_plot.txt')
plt.figure(1)
plt.clf()
plt.xlabel('height (m)')
plt.grid()
plt.plot(results[:, 0], results[:, 1], label='radial speed (m/s)')
plt.legend()
plt.show()