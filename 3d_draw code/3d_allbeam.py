from astropy.io import fits
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import matplotlib.ticker as ticker
import random
from mpl_toolkits.mplot3d import Axes3D
import time

t1 = time.time()

# rui update:April 15th

# read data

# modify this line
# data.shape = (19,256,1573) 19->beam 256->time 1573->freq
localpath = '168_0_all.fits'

hdu1 = fits.open(localpath)

data = hdu1[0].data
arr = data
z, x, y = np.nonzero(arr)
# x,y,z.shape = (7651072,) = (19*256*1573)

# color with flux
color = np.reshape(arr, (7651072,))

# change freq to 1300~1420
y = y / 1.00000000
y.astype(float)
# generate freq_HI
freq_all = np.arange(65536)*0.00762939+1000.00357628
freq_all.shape
indfreq_HI = np.where((freq_all < 1420) & (freq_all > 1300))[0]
indfreq_HI.shape
freq_HI = freq_all[indfreq_HI]
freq_HI
freq_HI_ad = np.zeros(1573,)
for i in range(0, 1573):
    freq_HI_ad[i] = freq_HI[10 * i]
# 4864 = 19*256
for j in range(0, 4864):
    for i in range(0, 1573):
        y[1573*j+i] = freq_HI_ad[i]

# draw
fig = plt.figure(figsize=(200, 200))
axes3d = Axes3D(fig)
ax = fig.gca(projection='3d')

sc = ax.scatter(x, y, -z, zdir='z', c=color)
ax.grid(False)
cb = plt.colorbar(sc)
cb.ax.tick_params(labelsize=10)
axes3d.set_zlabel('beam', fontsize=10)
axes3d.set_xlabel('time', fontsize=10)
axes3d.set_ylabel('freq', fontsize=10)

plt.xlim(0, 256)
plt.ylim(1300, 1420)

tick_spacing_x = 8
ax.xaxis.set_minor_locator(ticker.MultipleLocator(tick_spacing_x))
tick_spacing_z = 1
ax.zaxis.set_minor_locator(ticker.MultipleLocator(tick_spacing_z))

plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
# plt.zticks(fontsize=5)

ax.w_xaxis.set_pane_color((1, 1, 1, 0.5))
ax.w_yaxis.set_pane_color((1, 1, 1, 0.5))
ax.w_zaxis.set_pane_color((1, 1, 1, 0.5))

plt.savefig(localpath+'1'+'.jpg')
t2 = time.time()
print(t2-t1)
# plt.show()
