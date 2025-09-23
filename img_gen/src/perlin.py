import math
import matplotlib.pyplot as plt
import numpy as np

def get_location_in_grid(x, y):
    gridsize = 1.0

    # get the grid index
    gridx = math.floor(x / gridsize)
    gridy = math.floor(y / gridsize)
    x_rem = (x % gridsize) / gridsize
    y_rem = (y % gridsize) / gridsize

    return gridx, gridy, x_rem, y_rem


def perlin(x, y):

    gx, gy, rx, ry = get_location_in_grid(x, y)
    location = np.array([rx, ry])

    g00 = get_gradient(gx, gy)
    g10 = get_gradient(gx+1, gy)
    g01 = get_gradient(gx, gy+1)
    g11 = get_gradient(gx+1, gy+1)

    # distance within the grid
    d00 = np.array([-location[0], -location[1]])
    d10 = np.array([1 - location[0], -location[1]])
    d01 = np.array([-location[0], 1 - location[1]])
    d11 = np.array([1 - location[0], 1 - location[1]])

    # dot products
    dot00 = np.dot(g00, d00)
    dot10 = np.dot(g10, d10)
    dot01 = np.dot(g01, d01)
    dot11 = np.dot(g11, d11)

    u = smooth(rx)
    v = smooth(ry)

    x1 = lerp(dot00, dot10, u)
    x2 = lerp(dot01, dot11, u)

    val = lerp(x1, x2, v)

    return np.float64(val)

def lerp(x1, x2, y):
    return (1-y)*x1 + (y)*x2


def smooth(val):
    return 6*math.pow(val, 5) - 15*math.pow(val, 4) + 10*math.pow(val, 3)


def get_gradient(x, y):
    # ensure values are correct type
    ix = np.int32(x)
    iy = np.int32(y)

    hash = ix
    hash ^= iy * 1609587929
    #hash ^= iy * 1209587929
    hash = hash*9650029 + 7
    hash = (hash >> 13) ^ hash

    angle = np.float64(hash&7) * math.pi / 4.0
    return np.array([math.cos(angle), math.sin(angle)])


def perlin_octaves(x, y, n):
    val = 0.0
    freq = 1.0
    amplitude = 1.0
    tot_amp = 0.0
    for ii in range(int(n)): 
#        val += amplitude * (0.5 + 0.5*(perlin(x*freq, y*freq)/0.60722))
        val += amplitude * perlin(x*freq, y*freq)
        tot_amp += amplitude
        freq *= 2
        amplitude /= 2

    return val / tot_amp + 0.5


if __name__ == "__main__":
    fig, ax = plt.subplots()
    arr = np.zeros([400, 400])
    xoffset = 2**15*np.random.rand()
    yoffset = 2**15*np.random.rand()
    for ii in range(400):
        for jj in range(400):
            arr[ii][jj] = perlin_octaves(ii/400.0+xoffset, jj/400.0+xoffset, 8)

    ax.imshow(arr, interpolation='nearest')
    print(np.min(arr), np.max(arr))
    plt.show()
