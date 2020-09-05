import numpy as np
from arguments import args
from tools.utils import RunningMeanStd


def gen_random_tsp(n_nodes):
    coords = []
    disMat = np.zeros((n_nodes, n_nodes), dtype=np.float)

    num = 0
    while num < n_nodes:
        tag = False
        x_coord, y_coord = np.random.rand(), np.random.rand()
        for coord in coords:
            if x_coord == coord[0] and y_coord == coord[1]:
                tag = True
        if not tag:
            coords.append([x_coord, y_coord])
            num += 1
        else:
            continue
    coords = np.array(coords)

    for i in range(n_nodes):
        coord = coords[i]
        disMat[i] = np.sum((coord - coords)**2, axis=1)**0.5

    return coords, disMat


if __name__ == '__main__':
    coords_ = gen_random_tsp(10)
    print(coords_)
