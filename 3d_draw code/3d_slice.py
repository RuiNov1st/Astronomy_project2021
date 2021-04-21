from astropy.io import fits
import numpy as np
# draw signal slice
# whole data is too big
# data.shape = (19,256,1573) 256->time 1573->freq

# modify this line
localpath = '/mnt/data66/home/weishirui/Documents/astropy_data/168_0_all.fits'
hdu1 = fits.open(localpath)
# hdu1.info()

# choose the signal you wanna draw
# signal 168:
# M1 index:(10758 890)->1382.08342748
# slice:freq->(1370,1390) time->(500,1100) beam->(1-4)
sig_freq_index = 1382
sig_time_index = 890

from astropy.io import fits
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import matplotlib.ticker as ticker
import random
from mpl_toolkits.mplot3d import Axes3D
import time

# timeindex+-200

def getTime(index):
    time = []
    # <left,right>:index+-200
    left = index-200
    if(left<0):
        left=0
    right = index+200
    if(right>2048):
        right = 2048
    flag = 1
    time_arr = np.arange(0,2048,8)
    for i in range(0,len(time_arr)):
        if(time_arr[i]>=left and flag==1):
            time.append(i)
            flag = 2
        else:
            if(time_arr[i]>=right and flag==2):
                time.append(i)
                break
    return time
time = getTime(sig_time_index)

# freq exchange
freq_all = np.arange(65536)*0.00762939+1000.00357628
#freq_all.shape
indfreq_HI = np.where((freq_all<1420)&(freq_all>1300))[0]
#indfreq_HI.shape
freq_HI = freq_all[indfreq_HI]
#freq_HI.shape
freq_HI_ad = np.zeros(1573,)
for i in range(0,1573):
    freq_HI_ad[i] = freq_HI[10*i]
#freq_HI_ad

# freqindex +-5
def getFreq(index):
    freq = []
    #<left,right>:index+-5
    left = index-5
    right = index+5
    if(left<0):
        left = 0
    if(right<0):
        right = 0
    flag = 1
    for i in range(0,1573):
        if(freq_HI_ad[i]>=left and flag==1):
            freq.append(i)
            flag = 2
        if(freq_HI_ad[i]>=right and flag==2):
            freq.append(i)
            break;
    return freq
freqindex = getFreq(sig_freq_index)
# print(freqindex[0],freqindex[1])


# read data
data = hdu1[0].data
arr = data
# beam:1-4,time->(690,1090),freq->(1377,1387)
arr1 = arr[0:4,time[0]:time[1],freqindex[0]:freqindex[1]]
# arr1.shape
#arr1
beam_shape = 4
time_shape = time[1]-time[0]
freq_shape = freqindex[1] - freqindex[0]
#print(beam_shape,time_shape,freq_shape)
#arr1.shape
z,x,y = np.nonzero(arr1)
color = np.reshape(arr1,x.shape)
#x.shape
y = y/1.00000000
y.astype(float)
#x.shape

# switch y into specific index
for j in range(0,beam_shape*time_shape):
    for i in range(0,freq_shape):
        y[freq_shape*j+i] =freq_HI_ad[freqindex[0]+i]
#y

# draw

#t1 = time.time()
fig = plt.figure(figsize = (200,200))
axes3d = Axes3D(fig)
ax = fig.gca(projection = '3d')


sc = ax.scatter(x,y,-z,zdir = 'z',c = color)
ax.grid(False)
plt.colorbar(sc)
axes3d.set_zlabel('beam')
axes3d.set_xlabel('time')
axes3d.set_ylabel('freq')
#plt.xlim(63,139)
plt.ylim(sig_freq_index-5,sig_freq_index+5)
#plt.yticks(freq_HI_ad)
tick_spacing_x = 8
ax.xaxis.set_minor_locator(ticker.MultipleLocator(tick_spacing_x))
tick_spacing_z = 1
ax.zaxis.set_minor_locator(ticker.MultipleLocator(tick_spacing_z))
#tick_spacing_y = 

ax.w_xaxis.set_pane_color((0,0,0,0.5))
ax.w_yaxis.set_pane_color((0,0,0,0.5))
ax.w_zaxis.set_pane_color((0,0,0,0.5))

#fig = plt.figure()
#plt.title("beam"+str(2))

plt.savefig(localpath+'1234'+'slice'+'.jpg')
#t2 = time.time()
#print(t2-t1)

plt.show()

