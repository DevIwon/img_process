import numpy as np ,cv2
import math
from math import log,log10,log1p
from Common.dft2d import dft, idft, calc_spectrum, fftshift
from Common.dct2d import dct,idct
import matplotlib.pylab as pylab

image=cv2.imread("./boat.jpg",cv2.IMREAD_GRAYSCALE)
if image is None: raise Exception("영상파일 오류")

def watermark(t):
    #Multi
    for i in range(5):
        t_p=t.flatten()+(alpha*abs(t.flatten())*y[(i+1)*100])
        t=t_p
    # t_p = t.flatten() + (alpha * abs(t.flatten()) * y[ 100])
    return t_p
def img_process():
    idx=0
    for i in range(175, 200):
        for j in range(50, 200):
            dct_img[i][j] = wm_concat[idx]
            idx += 1
    for i in range(50, 175):
        for j in range(175, 200):
            dct_img[i][j] = wm_concat[idx]
            idx += 1

def wm_detect(detect,y):
    Z=np.zeros(1000)
    for i in range(1000):
        Z[i]=np.mean(detect*y[i,:])
    return Z

def low_pass(wm_img):
    gss_img = wm_img.copy()
    gss_detect = cv2.GaussianBlur(gss_img, (5, 5),0)
    gss=dct(gss_detect)
    gss_temp=gss[175:200,50:200].flatten()
    gss_temp2=gss[50:175,175:200].flatten()
    gss_concat=np.concatenate([gss_temp,gss_temp2])

    return gss_concat

def show(wm_concat,gss_concat):
    pylab.figure(figsize=(10, 10))
    pylab.subplot(1, 2, 1), pylab.title("original"), pylab.imshow(image, cmap='gray')
    pylab.subplot(1, 2, 2), pylab.title("Fig2"), pylab.imshow(wm_img, cmap='gray')
    pylab.show()
    pylab.figure(figsize=(10, 10))
    pylab.subplot(2, 1, 1), pylab.title("Fig3,14"), pylab.plot(wm_result)
    pylab.axhline((alpha / (2 * len(wm_concat))) * np.sum(abs(wm_concat)), color='lightgray', linestyle='--')
    pylab.subplot(2, 1, 2), pylab.title("Fig5"), pylab.plot(gss_result)
    pylab.axhline((alpha / (3 * len(gss_concat))) * np.sum(abs(gss_concat)), color='lightgray', linestyle='--')
    pylab.show()

alpha=0.3
y=np.random.randn(1000,6875)
dct_img=dct(image)
dct_copy=dct_img.copy()


wm_temp=dct_copy[175:200,50:200].flatten()
wm_temp2=dct_copy[50:175,175:200].flatten()

wm_concat=np.concatenate([wm_temp,wm_temp2])
wm_concat=watermark(wm_concat)
img_process()

wm_result=wm_detect(wm_concat,y)
wm_img=idct(dct_img)
gss_concat=low_pass(wm_img)
gss_result = wm_detect(gss_concat, y)
show(wm_concat,gss_concat)
