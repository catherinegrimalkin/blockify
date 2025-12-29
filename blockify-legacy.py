#important note: this code expects images which have already been vertically squished by 50%.
#if one is using imagemagick to convert one's pictures to the PGM P6 image format this code also expects...
#that's something one can add to that invocation.
#otherwise: sorry!

import sys
import numpy as np

blocks = {"1111111111111111111111111111111100000000000000000000000000000000": "▀", 
          "0000000000000000000000000000000000000000000000000000000011111111": "▁", 
          "0000000000000000000000000000000000000000000000001111111111111111": "▂", 
          "0000000000000000000000000000000000000000111111111111111111111111": "▃", 
          "0000000000000000000000001111111111111111111111111111111111111111": "▅", 
          "0000000000000000111111111111111111111111111111111111111111111111": "▆", 
          "0000000011111111111111111111111111111111111111111111111111111111": "▇", 
          "1111111011111110111111101111111011111110111111101111111011111110": "▉", 
          "1111111011111100111111001111110011111100111111001111110011111100": "▊", 
          "1111100011111000111110001111100011111000111110001111100011111000": "\u258b", 
          "1111000011110000111100001111000011110000111100001111000011110000": "\u258c", 
          "1110000011100000111000001110000011100000111000001110000011100000": "\u258d", 
          "1100000011000000110000001100000011000000110000001100000011000000": "▍", 
          "1000000010000000100000001000000010000000100000001000000010000000": "▏", 
          "1111000011110000111100001111000000000000000000000000000000000000": "▘", 
          "1111000011110000111100001111000011111111111111111111111111111111": "▙", 
          "1111000011110000111100001111000000001111000011110000111100001111": "▚", 
          "1111111111111111111111111111111111110000111100001111000011110000": "▛", 
          "1111111111111111111111111111111100001111000011110000111100001111": "▜", 
          "0000000000000000000000001111111111111111000000000000000000000000": "━",
          "0001100000011000000110000001100000011000000110000001100000011000": "┃", 
          "0000000000000000000000000001111100011111000110000001100000011000": "┏", 
          "0000000000000000000000001111100011111000000110000001100000011000": "┓",
          "0001100000011000000110000001111100011111000000000000000000000000": "┗", 
          "0001100000011000000110001111100011111000000000000000000000000000": "┛", 
          "0001100000011000000110000001111100011111000110000001100000011000": "┣", 
          "0001100000011000000110001111100011111000000110000001100000011000": "┫", 
          "0000000000000000000000001111111111111111000110000001100000011000": "┳", 
          "0001100000011000000110001111111111111111000000000000000000000000": "┻", 
          "0001100000011000000110001111111111111111000110000001100000011000": "╋",
          "0000000000000000000000001111100011111000000000000000000000000000":"╸",
          "0001100000011000000110000001100000011000000000000000000000000000":"╹",
          "0000000000000000000000000001111100011111000000000000000000000000":"╺",
          "0000000000000000000000000001100000011000000110000001100000011000":"╻",
          "0000000000000000011111100111111001111110011111100000000000000000":"▪",
          "0000000000000000011111100100001001000010011111100000000000000000":"▫",
          "0000000000000000000000000001100000011000000000000000000000000000":"·",}


#define an arbitrary ordering for the blocks, so they can be stored by-index later
keys = tuple(blocks.keys())

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

#read PNM P6 file from stdin (breaks if file has a comment, but, eh)
assert sys.stdin.buffer.readline() == b"P6\n"
w, h = map(int, sys.stdin.buffer.readline().split())
depth = int(sys.stdin.buffer.readline())
a = np.array(list(map(int, sys.stdin.buffer.read())))

#convert from sRGB to okLAB
b = a / depth
b = np.vectorize(f_inv)(b)
b = b.reshape(h, w, 3)
b = b@tolab_1.transpose()
b **= 1 / 3
b = b@tolab_2.transpose()

#height and width of each block
hs = 8
ws = 8

#the crux of the algorithm is using numpy vectorization to calculate how well a given block fits for *every* chunk of the image at the same time
#so these are, respectively:
#the current best score for each chunk
#the best foreground color for each chunk
#the best background color for each chunk
#and the index of the best block for each chunk
bests = np.ones((h // hs, w // ws), dtype=float) * float("inf")
c1 = np.zeros((h // hs, w // ws, 3), dtype=float)
c2 = np.zeros((h // hs, w // ws, 3), dtype=float)
c3 = np.zeros((h // hs, w // ws), dtype=int)

for bl in range(len(keys)):
    #find the foreground color and background color for this block, using the arithmetic mean, for every chunk at once 
    icnt = 0
    jcnt = 0
    i = np.zeros((h // hs, w // ws, 3), dtype=float)
    j = np.zeros((h // hs, w // ws, 3), dtype=float)
    for y_ in range(hs):
        for x_ in range(ws):
            if int(keys[bl][y_ * ws + x_]):
                icnt += 1
                i += b[y_::hs, x_::ws]
            else:
                jcnt += 1
                j += b[y_::hs, x_::ws]
    i /= icnt
    j /= jcnt
    
    #find the total euclidean-distance deviation from the image, for this block, for every chunk at once
    scores = np.zeros((h // hs, w // ws), dtype=float)
    for y_ in range(hs):
        for x_ in range(ws):
            errs = (b[y_::hs, x_::ws] - (i if int(keys[bl][y_ * ws + x_]) else j)) ** 2
            scores += (errs[:, :, 0] + errs[:, :, 1] + errs[:, :, 2]) ** 0.5

    #numpy broadcasting hack: find the chunks where this block is better than all the ones we've tried before...
    wins = scores < bests
    #...and update our best-block tracker on *just* those
    bests[wins] = scores[wins]
    c1[wins] = i[wins]
    c2[wins] = j[wins]
    c3[wins] = bl

#once the loop is finished, it should have checked, and found, the block with the minimum deviation for every chunk of the image.
#now all that's left is to act on this info: first, converting all the values from okLAB back to sRGB
c1 = c1@fromlab_1.transpose()
c1 **= 3
c1 = c1@fromlab_2.transpose()
c1 = np.vectorize(f)(c1)
c1 *= 255

c2 = c2@fromlab_1.transpose()
c2 **= 3
c2 = c2@fromlab_2.transpose()
c2 = np.vectorize(f)(c2)
c2 *= 255

#and then printing each block as calculated.
for y in range(h // hs):
    for x in range(w // ws):
        g1 = c1[y, x]
        g2 = c2[y, x]
        g3 = keys[c3[y, x]]
        print(f"\x1b[38;2;{int(g1[0])};{int(g1[1])};{int(g1[2])};48;2;{int(g2[0])};{int(g2[1])};{int(g2[2])}m{blocks[g3]}", end="")
    print("\x1b[0m")
