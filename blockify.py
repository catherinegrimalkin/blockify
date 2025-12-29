import sys
import numpy as np
from math import ceil
from sys import argv
import unicodedata
from wcwidth import wcswidth

def f_inv(x):
    if (x >= 0.04045):
        return ((x + 0.055)/(1 + 0.055)) ** 2.4
    else:
        return x / 12.92

def f(x):
    if (x >= 0.0031308):
        return (1.055) * x ** (1.0/2.4) - 0.055
    else:
        return 12.92 * x

tolab_1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929], 
                    [0.2119034982, 0.6806995451, 0.1073969566], 
                    [0.0883024619, 0.2817188376, 0.6299787005]])
tolab_2 = np.array([[0.2104542553, +0.7936177850, -0.0040720468], 
                    [1.9779984951, -2.4285922050, +0.4505937099], 
                    [0.0259040371, +0.7827717662, -0.8086757660]])

fromlab_1 = np.array([[1, 0.3963377774, +0.2158037573], 
                    [1, -0.1055613458, -0.0638541728], 
                    [1, 0.0894841775, - 1.2914855480]])
fromlab_2 = np.array([[+4.0767416621, -3.3077115913, +0.2309699292], 
		      [-1.2684380046, +2.6097574011, -0.3413193965],
		      [-0.0041960863, -0.7034186147, +1.7076147010]])

hs = 16
ws = 8

with open(argv[1], "rb") as inp:
    assert inp.readline() == b"P6\n"
    w, h = map(int, inp.readline().split())
    depth = int(inp.readline())
    a = np.array(list(map(int, inp.read())))
b = a / depth
b = np.vectorize(f_inv)(b)
b = b.reshape(h, w, 3)
if h < 500:
    b = np.repeat(b, 2, 0)
    b = np.repeat(b, 2, 1)
    w *= 2
    h *= 2
scaler = max(ceil(w / (144 * ws)), ceil(h / (44 * hs)))
if scaler:
    b =b[::scaler, ::scaler, :]
h = len(b)
w = len(b[0])
new_h = ceil(h / hs) * hs
new_w = ceil(w / ws) * ws
b = np.pad(b, ((0, new_h - h), (0, new_w - w), (0, 0)))
h = new_h
w = new_w
b = b@tolab_1.transpose()
b **= 1 / 3
b = b@tolab_2.transpose()

bests = np.ones((h // hs, w // ws), dtype=float) * float("inf")
c1 = np.zeros((h // hs, w // ws, 3), dtype=float)
c2 = np.zeros((h // hs, w // ws, 3), dtype=float)
c3 = np.zeros((h // hs, w // ws), dtype=np.ulonglong)
cnt = 0
for unif in sys.stdin:
    if len(unif) > 64:
        continue
    bl = int(unif.split(":")[0], 16)
    bl2 = int(unif.split(":")[1], 16)
    if bl2 == 0:
        continue
#    if (0x0300 <= bl <= 0x036F):
#        continue
#    if chr(bl) in "⚫◾":
    if unicodedata.category(chr(bl))[0] == "M":
        continue
    if wcswidth(chr(bl) + "\uFE0E") > 1:
        continue
    print(chr(bl) + unif, file=sys.stderr, end="")
#    print(bl, bl2)
    icnt = 0
    jcnt = 0
    i = np.zeros((h // hs, w // ws, 3), dtype=float)
    j = np.zeros((h // hs, w // ws, 3), dtype=float)
    for y_ in range(hs - 1, -1, -1):
        for x_ in range(ws - 1, -1, -1):
            if bl2 % 2:
                icnt += 1
                i += b[y_::hs, x_::ws]
            else:
                jcnt += 1
                j += b[y_::hs, x_::ws]
            bl2 //= 2
    if icnt == 0 or jcnt == 0:
        continue
    i /= icnt
    j /= jcnt
    scores = np.zeros((h // hs, w // ws), dtype=float)
    bl2 = int(unif.split(":")[1], 16)
    for y_ in range(hs - 1, -1, -1):
        for x_ in range(ws - 1, -1, -1):
            if bl2 % 2:
                errs = (b[y_::hs, x_::ws] - i) ** 2
            else:
                errs = (b[y_::hs, x_::ws] - j) ** 2
            bl2 //= 2
            scores += (errs[:, :, 0] + errs[:, :, 1] + errs[:, :, 2])
    wins = scores < bests
    bests[wins] = scores[wins]
    c1[wins] = i[wins]
    c2[wins] = j[wins]
    c3[wins] = bl
    cnt += 1
    if (cnt % 1000) == 0:
        c1copy = c1@fromlab_1.transpose()
        c1copy **= 3
        c1copy = c1copy@fromlab_2.transpose()
        c1copy = np.vectorize(f)(c1copy)
        c1copy *= 255
        c1copy = np.maximum(0, np.minimum(c1copy, 255))

        c2copy = c2@fromlab_1.transpose()
        c2copy **= 3
        c2copy = c2copy@fromlab_2.transpose()
        c2copy = np.vectorize(f)(c2copy)
        c2copy *= 255
        c2copy = np.maximum(0, np.minimum(c2copy, 255))
        for y in range(h // hs):
            for x in range(w // ws):
                g1 = c1copy[y, x]
                g2 = c2copy[y, x]
                g3 = chr(c3[y, x]) + "\uFE0E"
                if wcswidth(g3) == 0:
                    g3 = " " + g3
                print(f"\x1b[38;2;{int(g1[0])};{int(g1[1])};{int(g1[2])};48;2;{int(g2[0])};{int(g2[1])};{int(g2[2])}m{g3}", end="", file=sys.stderr)
            print("\x1b[0m", file=sys.stderr)

        

c1 = c1@fromlab_1.transpose()
c1 **= 3
c1 = c1@fromlab_2.transpose()
c1 = np.vectorize(f)(c1)
c1 *= 255
c1 = np.maximum(0, np.minimum(c1, 255))

c2 = c2@fromlab_1.transpose()
c2 **= 3
c2 = c2@fromlab_2.transpose()
c2 = np.vectorize(f)(c2)
c2 *= 255
c2 = np.maximum(0, np.minimum(c2, 255))
for y in range(h // hs):
    for x in range(w // ws):
        g1 = c1[y, x]
        g2 = c2[y, x]
        g3 = chr(c3[y, x]) + "\uFE0E"
        if wcswidth(g3) == 0:
            g3 = " " + g3
        print(f"\x1b[38;2;{int(g1[0])};{int(g1[1])};{int(g1[2])};48;2;{int(g2[0])};{int(g2[1])};{int(g2[2])}m{g3}", end="")
#        print("\x1b[0m ", end="")
    print("\x1b[0m")
