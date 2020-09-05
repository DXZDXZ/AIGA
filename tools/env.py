import random
import numpy as np

from tools.utils import get_chromo_distance, get_distance_two_nodes


class TSP_node(object):
    def __init__(self, x, y, px, py, dis, embedding=None):
        self.x = x
        self.y = y
        self.px = px
        self.py = py
        self.dis = dis
        if embedding is None:
            self.embedding = [self.x, self.y, self.px, self.py, self.dis]
        else:
            self.embedding = embedding.copy()


class Chromosome_Env:
    def __init__(self, seq, disMat):
        self.seq = seq

    def get_states(self):
        pass


class TSP_env:
    def __init__(self, n_nodes, coords, disMat, pop_size):
        self.n_nodes = n_nodes  # 客户点节点个数，同时也是每个个体的长度，即基因长度
        self.coords = coords  # 客户点坐标
        self.disMat = disMat  # 客户点距离矩阵
        self.pop_size = pop_size  # 种群大小

        self.best_chromosome = None  # 最优染色体，初始化为第 0 个染色体
        self.best_chromosome_length = None  # 最优解的长度
        self.encoder_outputs = None

        self.edge_index_n = self.get_edge_index_n(n_nearest=self.n_nodes)
        self.nodes_embedding = self.get_nodes_embedding()
        self.chromosomes = self.init_population()  # 当前种群
        self.routes, self.tot_dis = self.get_routes_totdis()
        self.fitness = self.get_fitness()

    def get_edge_index_n(self, n_nearest, inverse=False):
        froms, tos = [], []
        for i in range(self.n_nodes):
            for node in np.argsort(self.disMat[i])[1:n_nearest + 1]:
                if inverse:
                    tos.append(node)
                    froms.append(i)
                else:
                    froms.append(node)
                    tos.append(i)
        return np.array([froms, tos])

    def get_nodes_embedding(self):
        nodes_embedding = []
        for i in range(self.n_nodes):
            new_node = TSP_node(x=self.coords[i][0],
                                y=self.coords[i][1],
                                px=self.coords[i][0],
                                py=self.coords[i][1],
                                dis=0.0)
            new_node.embedding = [new_node.x, new_node.y, new_node.px, new_node.py, new_node.dis]
            nodes_embedding.append(new_node.embedding)

        return nodes_embedding

    def get_routes_totdis(self):
        routes = []
        tot_dis = []

        for chromosome in self.chromosomes:
            route = []
            route_dis = []
            start_node = TSP_node(x=self.coords[chromosome[0]][0],
                                  y=self.coords[chromosome[0]][1],
                                  px=self.coords[chromosome[-1]][0],
                                  py=self.coords[chromosome[-1]][1],
                                  dis=get_distance_two_nodes(self.coords[chromosome[0]],
                                                             self.coords[chromosome[-1]]))
            start_node.embedding = [start_node.x, start_node.y,
                                    start_node.px, start_node.py,
                                    start_node.dis]
            route.append(start_node.embedding[:])
            route_dis.append(start_node.dis)

            for c in range(1, len(chromosome)):
                new_node = TSP_node(x=self.coords[chromosome[c]][0],
                                    y=self.coords[chromosome[c]][1],
                                    px=self.coords[chromosome[c - 1]][0],
                                    py=self.coords[chromosome[c - 1]][1],
                                    dis=get_distance_two_nodes(self.coords[chromosome[c]],
                                                               self.coords[chromosome[c - 1]]))
                new_node.embedding = [new_node.x, new_node.y,
                                      new_node.px, new_node.py,
                                      new_node.dis]
                route.append(new_node.embedding[:])
                route_dis.append(route_dis[-1] + new_node.dis)
            routes.append(route)
            tot_dis.append(route_dis)

        self.routes = routes
        self.tot_dis = tot_dis

        return routes, tot_dis

    def init_population(self):
        chromosomes = []
        for i in range(self.pop_size):
            chromosome = list(range(self.n_nodes))
            random.shuffle(chromosome)
            s_idx = chromosome.index(0)
            chromosome = chromosome[s_idx:] + chromosome[:s_idx]
            chromosomes.append(chromosome)

        return chromosomes

    def get_best_chromosome(self):
        for i in range(1, self.pop_size):
            current_distance = get_chromo_distance(self.chromosomes[i], self.disMat)
            if current_distance < self.best_chromosome_length:
                self.best_chromosome = self.chromosomes[i]
                self.best_chromosome_length = current_distance

    def get_fitness(self):
        fitness = []
        for route in self.routes:
            fit = 1.0 / route[-1][-1]
            fitness.append(fit)
        return fitness

    def update_population(self, offspring):
        pass

