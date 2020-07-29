import cv2
import numpy as np

"""
四隅に低周波数があるため、
切り抜いた四隅を合成し、中央を低周波数にする
"""
def shift_dft(src, dst=None):
    if dst is None:
        dst = np.empty(src.shape, src.dtype)
    elif src.shape != dst.shape:
        raise ValueError("src and dst must have equal sizes")
    elif src.dtype != dst.dtype:
        raise TypeError("src and dst must have equal types")

    if src is dst:
        ret = np.empty(src.shape, src.dtype)
    else:
        ret = dst

    h, w = src.shape[:2]

    cx1 = cx2 = w/2
    cy1 = cy2 = h/2

    # 
    if w % 2 != 0:
        cx2 += 1
    if h % 2 != 0:
        cy2 += 1

    # 左上と右下の入れ替え
    ret[h-cy1:, w-cx1:] = src[0:cy1 , 0:cx1 ]   # 左上を右下に
    ret[0:cy2 , 0:cx2 ] = src[h-cy2:, w-cx2:]   # 右下を右上に
    # 左下と右上の入れ替え
    ret[0:cy2 , w-cx2:] = src[h-cy2:, 0:cx2 ]   # 左下を右上に
    ret[h-cy1:, 0:cx1 ] = src[0:cy1 , w-cx1:]   # 右上を左下に

    if src is dst:
        dst[:,:] = ret

    return dst

if __name__ == "__main__":
    im = cv2.imread('keita.jpg')

    # グレースケール変換
    im2 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    h, w = im2.shape[:2]

    realInput = im2.astype(np.float64)

    # DFT変換のサイズを計算
    dft_M = cv2.getOptimalDFTSize(w)
    dft_N = cv2.getOptimalDFTSize(h)

    # Aをdft_Aにコピーし、 pad dft_Aはゼロ
    dft_A = np.zeros((dft_N, dft_M, 2), dtype=np.float64)
    dft_A[:h, :w, 0] = realInput

    # no need to pad bottom part of dft_A with zeros because of
    # use of nonzeroRows parameter in cv2.dft()
    cv2.dft(dft_A, dst=dft_A, nonzeroRows=h)

    cv2.imshow("win", im)

    # 実数と虚数分解
    image_Re, image_Im = cv2.split(dft_A)

    # 二乗和して根を取り正にへ
    magnitude = cv2.sqrt(image_Re**2.0 + image_Im**2.0)

    # 対数を取る
    log_spectrum = cv2.log(1.0 + magnitude)

    # 四隅が低波数⇒中央が低波数になるよう組み換え
    shift_dft(log_spectrum, log_spectrum)

    # 正規化して可視化
    cv2.normalize(log_spectrum, log_spectrum, 0.0, 1.0, cv2.NORM_MINMAX)
    
    log_spectrum2=np.float32(log_spectrum)

    cv2.imshow("magnitude", log_spectrum2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()