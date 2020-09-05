import numpy as np


def get_chromo_distance(chromosome, disMat):
    distance = 0
    for i in range(len(chromosome)):
        if i == len(chromosome) - 1:
            source, end = chromosome[i], chromosome[0]
        else:
            source, end = chromosome[i], chromosome[i + 1]
        distance += disMat[source][end]
    return distance


def get_distance_two_nodes(coord_1, coord_2):
    return ((coord_1[0] - coord_2[0]) ** 2 + (coord_1[1] - coord_2[1]) ** 2) ** 0.5


def build_edge_index(chrom, inverse=False):
    froms, tos = [], []
    if not inverse:
        froms.append(chrom[-1])
        tos.append(chrom[0])
    else:
        froms.append(chrom[0])
        tos.append(chrom[-1])
    for c in range(len(chrom) - 1):
        if not inverse:
            froms.append(chrom[c])
            tos.append(chrom[c + 1])
        else:
            froms.append(chrom[c + 1])
            tos.append(chrom[c])
    edge_index = [froms, tos]

    return edge_index


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

    def __call__(self, x):
        self.update(x)
        return (x - self.mean) / np.sqrt(self.var)


if __name__ == '__main__':
    result = build_edge_index([3, 2, 5, 0, 1, 4], inverse=True)
    print(result)
