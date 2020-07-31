import cv2
import numpy as np
from matplotlib import pyplot as plt
 
img = cv2.imread('keita.jpg', 0)           
f = np.fft.fft2(img)                      #FFTを計算するための関数
f_shift = np.fft.fftshift(f)                
mag = 20 * np.log(np.abs(f_shift))          
 
rows, cols = img.shape                      
crow, ccol = int(rows / 2), int(cols / 2)  
mask = 30                                 
f_shift[crow-mask:crow+mask,
        ccol-mask:ccol+mask] = 0
 
f_ishift = np.fft.ifftshift(f_shift)      #直流成分の位置を画像の左上に
img_back = np.fft.ifft2(f_ishift)         #逆フーリエ変換を適用
img_back = np.abs(img_back)                
 
fig = plt.figure(figsize=(10, 3))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)
ax1.imshow(img, cmap='gray')
ax2.imshow(mag, cmap='gray')
ax3.imshow(img_back, cmap='gray')
ax1.axis('off')
ax2.axis('off')
ax3.axis('off')
plt.tight_layout()
plt.show()
plt.close()