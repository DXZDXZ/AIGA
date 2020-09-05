import torch
from torch_geometric.data import Data, DataLoader

from arguments import args
from lib.tsp_model import TSP_Model
from tools.env import TSP_env
from lib.operators.selector import perform_select
from tools.data_generator import gen_random_tsp
from tools.utils import build_edge_index


def create_model(CONF):
    model = TSP_Model(CONF)
    model.to(CONF.device)
    return model


def train(CONF):

    model = create_model(CONF)

    for i_epoch in range(CONF.n_epochs):
        n_nodes = [200, 100, 50, 25][i_epoch % 4]
        print("[Epoch: %d/%d]-> nodes = %d | max_ga_iter_steps = %d" % (i_epoch + 1, CONF.n_epochs, n_nodes, CONF.max_ga_iter_steps))

        ##############################################################################
        # generate problems
        env_batch = []
        for i_batch in range(CONF.batch_size):
            coords, disMat = gen_random_tsp(n_nodes)        # tsp problem: coords, disMat
            env = TSP_env(n_nodes, coords, disMat, CONF.pop_size)
            env.init_population()
            env_batch.append(env)

        ###############################################################################
        # Begin iteration
        for i_iter in range(CONF.max_ga_iter_steps):

            # Select chromosomes from population
            mini_selected_chrom_batch = []
            mini_selected_route_batch = []
            for i_batch in range(CONF.batch_size):
                mini_selected_idx = perform_select(env_batch[i_batch], CONF.n_select_chroms)
                mini_selected_chroms = [env_batch[i_batch].chromosomes[idx] for idx in mini_selected_idx]
                mini_selected_routes = [env_batch[i_batch].routes[idx] for idx in mini_selected_idx]
                mini_selected_chrom_batch.append(mini_selected_chroms)
                mini_selected_route_batch.append(mini_selected_routes)
            # shape: [batch_size, n_select_chroms, n_nodes]
            mini_selected_chrom_batch = torch.tensor(mini_selected_chrom_batch, dtype=torch.float32).to(CONF.device)
            # shape: [batch_size, n_select_chroms, n_nodes, node_dim]
            mini_selected_route_batch = torch.tensor(mini_selected_route_batch, dtype=torch.float32).to(CONF.device)

            # Input into model or named crossover
            for i_selected in range(0, CONF.n_select_chroms, 2):      # Perform "crossover" separately on selected chromosomes
                selected_chrom_batch = mini_selected_chrom_batch[:, i_selected, :].squeeze(1)
                selected_route_batch = mini_selected_route_batch[:, i_selected, :, :].squeeze(1)

                edge_index_r0_batch = []
                edge_index_r1_batch = []
                for i_batch in range(CONF.batch_size):
                    edge_index_r0_batch.append(build_edge_index(selected_chrom_batch[i_batch], inverse=False))
                    edge_index_r1_batch.append(build_edge_index(selected_chrom_batch[i_batch], inverse=True))

                dataset = []
                for i_batch in range(CONF.batch_size):
                    data = Data(x=torch.tensor(env_batch[i_batch].nodes_embedding, dtype=torch.float32).to(CONF.device),
                                edge_index_r0=torch.tensor(edge_index_r0_batch[i_batch]).to(CONF.device),
                                edge_index_r1=torch.tensor(edge_index_r1_batch[i_batch]).to(CONF.device),
                                edge_index_n=torch.tensor(env_batch[i_batch].edge_index_n).to(CONF.device))
                    dataset.append(data)
                loader = DataLoader(dataset, batch_size=CONF.batch_size, shuffle=False)

                model(env_batch, list(loader)[0], selected_route_batch, n_nodes)


if __name__ == '__main__':
    args = args()
    train(args)


