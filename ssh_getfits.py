# 局部加权回归减少数据点
import numpy as np
from scipy.optimize import leastsq
from astropy.io import fits
import matplotlib.pyplot as plt
import time

t1 = time.time()
# filepath
localpath = '/mnt/data66/home/weishirui/Documents/Data_cube/data_cube/Dec+2715_20200213_0165_0.fits'
hdu1 = fits.open(localpath)
# data
data = hdu1[0].data
arr = data
# arr.shape:(19,2048,15728)
# ax1 = beam
for u in range(0,19):
    z = arr[u,:,:] #shape-(2048,15728)
    # 8 time points->one point->256 groups
    z_11 = np.empty([256,1573])
    # one row
    freq = np.arange(0,15728,1)

    #print(fluxdensity)
    print(freq)
    print(z_11.shape)

    from astropy.io import fits
    p0 = np.array([0,0])
    result = np.array([])

    # Regression
    def func(p,x):
        k,b = p
        return k*x+b

    def error(p,x,y,x_0,sigma):
        return (func(p,x)-y)*np.exp(-np.abs(x-x_0)/sigma)

    def printimg(freq,fluxdensity,result):
        x_new = np.arange(0,15730,10)
                    # print:
        fig,ax = plt.subplots(2,1,figsize = (12,16))
        ax[0].plot(freq,fluxdensity,'k-')
        ax[1].plot(x_new,result,'k-')
        #fig,ax = plt.subplots(2,1,figsize = (12,16))
                    #ax[0].plot(freq,fluxdensity,'k-')
                    #ax[1].plot(x_new,result,'k-')
            #t2 = time.time()
        #plt.title('sigma = '+str(sigmoid[j])+' '+str(t2-t1))
            #localpath_save = '/mnt/data66/home/weishirui/Documents/Data_cube/decre_168_0_'+str(j)+'.jpg'
            #plt.savefig(localpath_save)
        plt.show()


    m = 0
    for j in range(0,2048,8):
        fluxdensity = z[j,:]/1e11
        #t1 = time.time()
        # sigmoid = [15,20,25,30,35,40]; 20 is OK
        for i in range(0,1573):
            # 10 points->1 points
            x_i = 10*i
            # sigmoid = 20
            para = leastsq(error,p0,args = (freq,fluxdensity,x_i,20))
            k,b = para[0]
            result = np.append(result,k*x_i+b)


        z_11[m] = result
        print(z_11[m])
       # printimg(freq,fluxdensity,result)
            #print(result)




        plt.show()
        result = np.array([])
        p0 = np.array([0,0])
        m+=1

    # write to new fits
    hdu2 = fits.PrimaryHDU(z_11)
    hdu3 = fits.HDUList([hdu2])
    hdu3.writeto('165_decre_'+str(u)+'.fits')
t2 = time.time()
print(t2-t1)
