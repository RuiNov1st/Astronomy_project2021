import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker


"""
fig = plt.figure(figsize=(100, 100))
axes3d = Axes3D(fig)
ax = fig.gca(projection='3d')
x = 0
y = 0
z = 1300
dx = 2048
dy = 19
dz = 120
xx = [x, x, x+dx, x+dx, x]
yy = [y, y+dy, y+dy, y, y]
kwargs = {'alpha': 0.3, 'color': 'gray'}
ax.plot3D(xx, yy, [z]*5, **kwargs)
ax.plot3D(xx, yy, [z+dz]*5, **kwargs)
ax.plot3D([x, x], [y, y], [z, z+dz], **kwargs)
ax.plot3D([x, x], [y+dy, y+dy], [z, z+dz], **kwargs)
ax.plot3D([x+dx, x+dx], [y+dy, y+dy], [z, z+dz], **kwargs)
ax.plot3D([x+dx, x+dx], [y, y], [z, z+dz], **kwargs)

# time&beam&freq
time = np.empty(200)
beam = np.empty(200)
freq = np.empty(200)
for i in range(0, 200):
    time[i] = random.randrange(0, 2048)
    beam[i] = random.randint(1, 20)
    freq[i] = random.randrange(1300, 1420)

sc = ax.scatter(time, beam, freq, c=np.random.rand(200))
# yticks = np.linspace(1, 19)
# tick_spacing = 30
tick_spacing = 1
# zticks = np.linspace(1300, 1420)
# ax.set_yticks(yticks)
# ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
# ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
# ax.zaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax.grid(None)
plt.colorbar(sc)
axes3d.set_xlabel('time')
axes3d.set_ylabel('beam')
axes3d.set_zlabel('freq')
ax.w_xaxis.set_pane_color((0, 0, 0, 0.5))
ax.w_yaxis.set_pane_color((0, 0, 0, 0.5))
ax.w_zaxis.set_pane_color((0, 0, 0, 0.5))


plt.show()
"""

"""
x = 0
y = 0
z = 1300
dx = 2048
dy = 19
dz = 120
xx = [x, x, x+dx, x+dx, x]
yy = [y, y+dy, y+dy, y, y]
kwargs = {'alpha': 0.3, 'color': 'gray'}
ax.plot3D(xx, yy, [z]*5, **kwargs)
ax.plot3D(xx, yy, [z+dz]*5, **kwargs)
ax.plot3D([x, x], [y, y], [z, z+dz], **kwargs)
ax.plot3D([x, x], [y+dy, y+dy], [z, z+dz], **kwargs)
ax.plot3D([x+dx, x+dx], [y+dy, y+dy], [z, z+dz], **kwargs)
ax.plot3D([x+dx, x+dx], [y, y], [z, z+dz], **kwargs)
"""


# time&beam&freq
largearray = np.random.randint(0, 100, size=(1, 10, 100))
z, x, y = largearray.nonzero()
fig = plt.figure(figsize=(100, 100))
axes3d = Axes3D(fig)
ax = fig.gca(projection='3d')
sc = ax.scatter(x, y, -z, zdir='z', vmin=0, vmax=6, c=np.random.rand(len(x)))


#ax = fig.add_subplot(111,projection='3d')
#ax.scatter(x, y, -z,zdir = 'z', c=np.random.rand(len(x)))
ax.grid(None)
#h = plt.contourf(largearray)
cb = plt.colorbar(sc)
cb.ax.tick_params(labelsize=10)
# plt.ticker.Locator(0.5)
# cb.set_ticks(tickss)
ax.spines['bottom'].set_linewidth(5)
axes3d.set_ylabel('time', fontsize=10)
axes3d.set_xlabel('beam', fontsize=10)
axes3d.set_zlabel('freq', fontsize=10)
tick_spacing = 1
ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

# plt.zticks(fontsize=5)
ax.w_xaxis.set_pane_color((0.6, 0.6, 0.6, 0.5))
ax.w_yaxis.set_pane_color((0.6, 0.6, 0.6, 0.5))
ax.w_zaxis.set_pane_color((0.6, 0.6, 0.6, 0.5))
# plt.savefig('1.jpg')
plt.show()
