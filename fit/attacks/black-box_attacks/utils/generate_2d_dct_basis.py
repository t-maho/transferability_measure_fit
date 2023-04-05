from math import cos, sqrt, pi
import numpy as np

def dct(x, y, v, u, n):
    # Normalisation
    def alpha(a):
        if a == 0:
            return sqrt(1.0 / n)
        else:
            return sqrt(2.0 / n)

    return alpha(u) * alpha(v) * cos(((2 * x + 1) * (u * pi)) / (2 * n)) * cos(((2 * y + 1) * (v * pi)) / (2 * n))


def generate_2d_dct_basis(image_size, sub_dim=75, save_path=None):
    # We can get different frequencies by setting u and v
    # Here, we have a max u and v to loop over and display
    # Feel free to adjust
    maxU = sub_dim
    maxV = sub_dim

    dct_basis = []
    for u in range(0, maxU):
        for v in range(0, maxV):
            basisImg = np.zeros(image_size)
            for y in range(0, image_size[0]):
                for x in range(0, image_size[1]):
                    basisImg[y, x] = dct(x, y, v, u, max(image_size[0], maxV))
            dct_basis.append(basisImg)
    dct_basis = np.mat(np.reshape(dct_basis, (maxV*maxU, image_size[0] * image_size[1]))).transpose()
    if save_path is not None:
        np.save(save_path, dct_basis)
    return dct_basis