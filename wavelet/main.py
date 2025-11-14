from typing import Tuple
import numpy as np

def haarDec(signal: np.ndarray, lb: np.ndarray, hb: np.ndarray) -> Tuple [np.ndarray, np.ndarray]: 
    a = np.convolve(signal, lb)
    d = np.convolve(signal, hb)
    a = a[1::2]
    d = d[1::2]
    return (a, d)


def simpleWaveDec(signal: np.ndarray, nb_scales: int) -> np.ndarray:
    C = np.empty(nb_scales + 1, dtype=object)
    a = signal
    lb = np.array([1, 1])
    hb = np.array([-1, 1])
    for scale in range(nb_scales):
        new_a, d = haarDec(a, lb, hb)
        a = new_a
        C[scale] = d
    C[nb_scales] = a
    return C

def reconstructScale(a: np.ndarray, d: np.ndarray, lr: np.ndarray, hr: np.ndarray) -> np.ndarray:
    over_a = np.zeros((len(a) * 2,))
    over_d = np.zeros((len(a) * 2,))

    over_a[::2] = a
    over_d[::2] = d

    over_a = np.convolve(over_a, lr)
    over_d = np.convolve(over_d, hr)

    rec = over_a + over_d
    rec = np.delete(rec, -1)
    return rec


def simpleWaveRec(C: np.ndarray) -> np.ndarray:
    lr = np.array([1, 1])/2
    hr = np.array([-1, 1])/2
    rec = C[-1]
    for detail in reversed(C[:-1]):
        rec = reconstructScale(rec, detail, lr, hr)
    return rec


data = np.array([4, 8, 2, 3, 5, 18, 19, 20], dtype=int)
dec = simpleWaveDec(data, 3)
rec = simpleWaveRec(dec)
print(rec)
