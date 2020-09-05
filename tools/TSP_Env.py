import random
import numpy as np


class Chromosome:
    def __init__(self, seq, coords, disMat):
        self.seq = seq
        self.disMat = disMat
        self.coords = coords

    def get_states(self):
        nodes = np.zeros((len(self.seq), 5))
        edges = np.zeros((len(self.seq), len(self.seq), 2))

        for i in range(len(self.seq)):
            if i == 0:
                pre_node_idx = self.seq[-1]
                node = [self.coords[i][0], self.coords[i][1],                           # 当前点的坐标
                        self.coords[pre_node_idx][0], self.coords[pre_node_idx][1],     # 前一个点的坐标
                        0]                                                              # 截止到当前点已走路程
            else:
                node_idx = self.seq[i]
                pre_node_idx = self.seq[i-1]
                node = [self.coords[i][0], self.coords[i][1],
                        self.coords[pre_node_idx][0], self.coords[pre_node_idx][1],
                        nodes[i][-1] + self.disMat[node_idx][pre_node_idx]]
            nodes[i] = node

        for l, r in zip(self.seq[0:-1], self.seq[1:]):
            edges[l][r][0] = self.disMat[l][r]
            edges[l][r][1] = 1
        edges = edges.reshape(-1, 2)

        return nodes, edges

    def get_length(self):
        length = 0.0
        for i in range(len(self.seq) - 1):
            length += self.disMat[self.seq[i]][self.seq[i+1]]
        return length


class TSP_Env(object):
    def __init__(self, n_nodes, coords, disMat, pop_size):
        self.n_nodes = n_nodes
        self.coords = coords
        self.disMat = disMat
        self.pop_size = pop_size
        self.chroms = [self.init_population() for _ in range(self.pop_size)]
        self.fits = [self.fitness(chrom) for chrom in self.chroms]
        self.edges = self.edge_index()
        self.mean_length = self.mean_cost()

    def init_population(self):
        seq = list(range(self.n_nodes))
        random.shuffle(seq)
        chrom = Chromosome(seq, self.coords, self.disMat)
        return chrom

    def fitness(self, chrom):
        fit = chrom.get_length()
        return 1.0 / fit

    def edge_index(self):
        edges = []
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                edges.append([i, j])
        return edges

    def mean_cost(self):
        tot_length = 0.0
        for chrom in self.chroms:
            tot_length += chrom.get_length()
        return tot_length / self.pop_size

    def create_chrom(self, seq):
        return Chromosome(seq, self.coords, self.disMat)

    def reset(self):
        self.chroms = [self.init_population() for _ in range(self.pop_size)]
        self.fits = [self.fitness(chrom) for chrom in self.chroms]
        self.edges = self.edge_index()
        self.mean_length = self.mean_cost()


    def step(self, chrom1_idx, chrom2_idx, actions1, actions2):
        chrom1 = self.chroms[chrom1_idx]
        chrom2 = self.chroms[chrom2_idx]
        actions1, _ = actions1.sort()
        actions2, _ = actions2.sort()
        pre_cost = (chrom1.get_length() + chrom2.get_length()) / 2.0
        # 对当前 Seq 进行修改，得到新的 states
        seq1, seq2 = chrom1.seq, chrom2.seq
        fragments1 = []
        fragments2 = []
        len_actions = len(actions1)
        for i in range(len_actions):
            if i == 0:
                fragments1.append(seq1[:actions1[i]])
                fragments2.append(seq2[:actions2[i]])
                continue
            fragments1.append(seq1[actions1[i-1] : actions1[i]])
            fragments2.append(seq2[actions2[i-1] : actions2[i]])
        fragments1.append(seq1[actions1[-1]:])
        fragments2.append(seq2[actions2[-1]:])

        rs = list(range(len_actions+1))
        random.shuffle(rs)
        new_seq1 = []
        new_seq2 = []
        for i in range(len_actions+1):
            new_seq1.extend(fragments1[rs[i]])
            new_seq2.extend(fragments2[rs[i]])
        new_chrom1 = Chromosome(new_seq1, self.coords, self.disMat)
        new_chrom2 = Chromosome(new_seq2, self.coords, self.disMat)

        l_index = np.argmin(self.fits)
        new_chrom1_fit = 1.0 / new_chrom1.get_length()

        if new_chrom1_fit > self.fits[l_index]:
            self.chroms[l_index] = new_chrom1
            self.fits[l_index] = new_chrom1_fit
            self.mean_length = self.mean_cost()

        l_index = np.argmin(self.fits)
        new_chrom2_fit = 1.0 / new_chrom2.get_length()

        if new_chrom2_fit > self.fits[l_index]:
            self.chroms[l_index] = new_chrom2
            self.fits[l_index] = new_chrom2_fit
            self.mean_length = self.mean_cost()

        new_length = min(new_chrom1.get_length(), new_chrom2.get_length())

        reward = pre_cost - new_length

        nodes1, edges1 = chrom1.get_states()
        nodes2, edges2 = chrom2.get_states()

        return nodes1, edges1, nodes2, edges2, reward


def create_env(n_jobs=99, _input=None):
    pass
