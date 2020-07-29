import wave
import numpy as np
import cmath as cm
import matplotlib.pyplot as plt

def wave_load(filename):
    # open wave file
    wf = wave.open(filename,'r')
    channels = wf.getnchannels() # 追記
    print(wf.getparams())

    # load wave data
    chunk_size = wf.getnframes()
    amp  = (2**8) ** wf.getsampwidth() / 2
    data = wf.readframes(chunk_size)   # バイナリ読み込み
    data = np.frombuffer(data,'int16') # intに変換
    data = data / amp                  # 振幅正規化

    return data


def DFT(data):
    res = []
    N = len(data)
    for k in range(N): #各周波数に関して
        w = cm.exp(-1j * 2 * cm.pi * k / float(N))
        X_k = 0
        for n in range(N): #信号*重みの総和をとる
            X_k += data[n] * (w ** n)
        res.append(abs(X_k))
    return res

if __name__ == '__main__':
    # 波形データ読み込み
    fs = 8000
    wave1 = wave_load('guitar_A4.wav')
    wave2 = wave_load('recorder_A4.wav')

    # DFT.時間かかるので一部のみを利用
    dt1 = DFT(wave1[10000:11024])
    dt2 = DFT(wave2[10000:11024])
    
    # 周波数リストを作成
    # 標本化周波数をデータ数で分割 
    frq = np.arange(1024) * fs / 1024

    # グラフ表示
    # ギターA4
    plt.subplot(2,1,1)
    plt.title('guitar_A4')
    plt.plot(frq,dt1)

    # リコーダーA4
    plt.subplot(2,1,2)
    plt.title('recorder_A4')
    plt.plot(frq,dt2)

    # グラフ表示
    plt.show()