from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


def main():
    # 画像を読み込む
    img = Image.open('keita.jpg')
    # グレイスケールに変換する
    gray_img = img.convert('L')
    # NumPy 配列にする
    f_xy = np.asarray(gray_img)

    # 2 次元高速フーリエ変換で周波数領域の情報を取り出す
    f_uv = np.fft.fft2(f_xy)
    # 画像の中心に低周波数の成分がくるように並べかえる
    shifted_f_uv = np.fft.fftshift(f_uv)

    # パワースペクトルに変換する
    magnitude_spectrum2d = 20 * np.log(np.absolute(shifted_f_uv))

    # 元の並びに直す
    unshifted_f_uv = np.fft.fftshift(shifted_f_uv)
    # 2 次元逆高速フーリエ変換で空間領域の情報に戻す
    i_f_xy = np.fft.ifft2(unshifted_f_uv).real  # 実数部だけ使う

    # 上記を画像として可視化する
    fig, axes = plt.subplots(1, 3, figsize=(8, 4))
    # 枠線と目盛りを消す
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
    # 元画像
    axes[0].imshow(f_xy, cmap='gray')
    axes[0].set_title('Input Image')
    # 周波数領域のパワースペクトル
    axes[1].imshow(magnitude_spectrum2d, cmap='gray')
    axes[1].set_title('Magnitude Spectrum')
    # FFT -> IFFT した画像
    axes[2].imshow(i_f_xy, cmap='gray')
    axes[2].set_title('Reversed Image')
    # グラフを表示する
    plt.show()


if __name__ == '__main__':
    main()