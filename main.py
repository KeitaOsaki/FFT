
	
import cv2
import numpy as np
from matplotlib import pyplot as plt
 
# 画像を読み込み、2Dフーリエ変換をする
img = cv2.imread('nice_face.jpg', 0)           # 画像をグレースケールで読み込み
f = np.fft.fft2(img)                        # 2Dフーリエ変換
f_shift = np.fft.fftshift(f)                # 直流成分を画像中心に移動させるためN/2シフトさせる
mag = 20 * np.log(np.abs(f_shift))          # 振幅成分を計算
 
# 周波数領域にマスクをかける
rows, cols = img.shape                      # 画像サイズを取得
crow, ccol = int(rows / 2), int(cols / 2)   # 画像中心を計算
mask = 30                                   # マスクのサイズ
f_shift[crow-mask:crow+mask,
        ccol-mask:ccol+mask] = 0
 
# 2D逆フーリエ変換によりフィルタリング後の画像を得る
f_ishift = np.fft.ifftshift(f_shift)        # シフト分を元に戻す
img_back = np.fft.ifft2(f_ishift)           # 逆フーリエ変換
img_back = np.abs(img_back)                 # 実部を計算する
 
# ここからグラフ表示
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