import lib
import numpy as np
import pylab as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

num_landmarks = 50
circ = lib.sample_circle(num_landmarks, scale=1, shift=0)
warp = lib.sample_circle(num_landmarks, scale=1, shift=0)
warp = np.array(map(lambda (x, y): [x+np.sin(-abs(y)**2)*2+1, y] if x<-0.2 else [x,y], warp))

fig = plt.figure(frameon=False)
ax = fig.add_axes([0, 0, 1, 1])
ax.axis('off')
lib.plot_landmarks(circ, color='r', start_style='o--', label='$\mathbf{Q}_0 = \{Q_0^i\}_{i=1}^M$', markersize=9)
#plt.legend(loc='best')
plt.savefig('circ.pdf', bbox_inches='tight')
plt.close()

fig = plt.figure(frameon=False)
ax = fig.add_axes([0, 0, 1, 1])
ax.axis('off')
lib.plot_landmarks(warp, start_style='x:', label='$\mathbf{Q}_1 = \{Q_1^i\}_{i=1}^M$', markersize=9)
plt.savefig('warp.pdf', bbox_inches='tight')
#plt.legend(loc='best')
plt.close()
