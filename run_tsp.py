import time
import random
import torch
import numpy as np
from torch_geometric.data import DataLoader

from arguments import args
from lib.tsp_model import TSP_Model
from tools.TSP_Env import TSP_Env
from lib.operators.selector import perform_select
from tools.data_generator import gen_random_tsp
from tools.replay_buffer import ReplayBuffer
from tools.utils import RunningMeanStd

reward_norm = RunningMeanStd()


def create_model(CONF):
    model = TSP_Model(CONF)
    model.to(CONF.device)
    # model.load_state_dict(torch.load("pretrained-model/4.model"))
    return model


def roll_out(CONF, model, envs, edges_index, chrom1_idx_batch, chrom2_idx_batch, nodes1, edges1, nodes2, edges2, n_steps,
             _lambda=0.99, is_last=False, greedy=False):
    buffer = ReplayBuffer()
    batch_size = len(envs)
    with torch.no_grad():
        model.eval()
        nodes1_batch, edges1_batch = nodes1, edges1
        nodes2_batch, edges2_batch = nodes2, edges2
        edges_index_batch = edges_index
        _entropy1 = []
        _entropy2 = []

        for i in range(n_steps):
            data1 = buffer.create_data(nodes1_batch, edges_index_batch, edges1_batch)
            data2 = buffer.create_data(nodes2_batch, edges_index_batch, edges2_batch)
            data1 = data1.to(CONF.device)
            data2 = data2.to(CONF.device)

            actions1, log_p1, values1, entropy1 = model(data1, 10, greedy)
            actions2, log_p2, values2, entropy2 = model(data2, 10, greedy)

            new_nodes1, new_nodes2, new_edges1, new_edges2, rewards = [], [], [], [], []
            for i_batch in range(len(envs)):
                env = envs[i_batch]
                new_node1, new_edge1, new_node2, new_edge2, reward = env.step(chrom1_idx_batch[i_batch],
                                                                              chrom2_idx_batch[i_batch],
                                                                              actions1[i_batch], actions2[i_batch])
                new_nodes1.append(new_node1)
                new_nodes2.append(new_node2)
                new_edges1.append(new_edge1)
                new_edges2.append(new_edge2)
                rewards.append(reward)

            rewards = np.array(rewards)
            rewards = reward_norm(rewards)
            _entropy1.append(entropy1)
            _entropy2.append(entropy2)

            buffer.obs(nodes1_batch, edges1_batch, actions1.cpu().numpy(), rewards, log_p1.cpu().numpy(),
                       values1.cpu().numpy())
            buffer.obs(nodes2_batch, edges2_batch, actions2.cpu().numpy(), rewards, log_p2.cpu().numpy(),
                       values2.cpu().numpy())

            nodes1_batch, edges1_batch = new_nodes1, new_edges1
            nodes2_batch, edges2_batch = new_nodes2, new_edges2

        if not is_last:
            data1 = buffer.create_data(nodes1_batch, edges_index_batch, edges1_batch)
            data2 = buffer.create_data(nodes2_batch, edges_index_batch, edges2_batch)
            data1 = data1.to(CONF.device)
            data2 = data2.to(CONF.device)
            actions1, log_p1, values1, entropy1 = model(data1, 10, greedy)
            actions2, log_p2, values2, entropy2 = model(data2, 10, greedy)
            values = values1.cpu().numpy() + values2.cpu().numpy()
        else:
            values = 0

        dl = buffer.gen_datas(values, edges_index_batch, batch_size=batch_size, _lambda=_lambda)

    return dl, (nodes1_batch, edges1_batch), (nodes2_batch, edges2_batch)


def train_once(CONF, model, opt, dl, epoch, step, alpha=1.0):
    model.train()

    losses = []
    loss_vs = []
    loss_ps = []
    _entropy = []

    for i, batch in enumerate(dl):
        batch = batch.to(CONF.device)
        batch_size = batch.num_graphs
        actions = batch.action.reshape((batch_size, -1))

        log_p, v, entropy = model.evaluate(batch, actions)

        _entropy.append(entropy.mean().item())

        target_vs = batch.v.squeeze(-1)
        old_log_p = batch.log_prob.squeeze(-1)
        adv = batch.adv.squeeze(-1)

        loss_v = ((v - target_vs) ** 2).mean()

        ratio = torch.exp(log_p - old_log_p)
        obj = ratio * adv
        obj_clipped = ratio.clamp(1.0 - 0.2, 1.0 + 0.2) * adv
        loss_p = -torch.min(obj, obj_clipped).mean()

        loss = loss_p + alpha * loss_v

        losses.append(loss.item())
        loss_vs.append(loss_v.item())
        loss_ps.append(loss_p.item())

        opt.zero_grad()
        loss.backward()
        opt.step()

    print("epoch:", epoch, "step:", step, "loss_v:", np.mean(loss_vs), "loss_p:", np.mean(loss_ps), "loss:",
          np.mean(losses), "entropy:", np.mean(_entropy))


def eval_random(CONF, n_nodes):
    env_batch = []
    for i_batch in range(CONF.batch_size):
        coords, disMat = gen_random_tsp(n_nodes)  # tsp problem: coords, disMat
        env = TSP_Env(n_nodes, coords, disMat, CONF.pop_size)
        env_batch.append(env)

    mean_length = np.mean([env.mean_length for env in env_batch])
    print("================= Eval Random | mean cost: %f =================" % mean_length)

    for i_iter in range(CONF.max_ga_iter_steps):

        # Select chromosomes from population
        selected_idxs_batch = []
        for i_batch in range(CONF.batch_size):
            selected_idxs = perform_select(env_batch[i_batch].fits, CONF.n_select_chroms)
            selected_idxs_batch.append(selected_idxs)
        selected_idxs_batch = np.array(selected_idxs_batch)

        for idx in range(0, CONF.n_select_chroms, 2):
            chrom1_idx_batch = selected_idxs_batch[:, idx]
            chrom1_idx_batch = np.array(chrom1_idx_batch)
            chrom2_idx_batch = selected_idxs_batch[:, idx + 1]
            chrom2_idx_batch = np.array(chrom2_idx_batch)

            actions1 = [random.sample(range(0, n_nodes), 10) for _ in range(CONF.batch_size)]
            actions2 = [random.sample(range(0, n_nodes), 10) for _ in range(CONF.batch_size)]
            actions1 = torch.tensor(actions1)
            actions2 = torch.tensor(actions2)
            for i_batch in range(CONF.batch_size):
                env = env_batch[i_batch]
                for i_step in range(CONF.n_rollout * CONF.roll_out_steps):
                    env.step(chrom1_idx_batch[i_batch],
                             chrom2_idx_batch[i_batch],
                             actions1[i_batch], actions2[i_batch])

        mmean_length = np.mean([env.mean_length for env in env_batch])
        print("Iter %d/%d | mean cost: %f" % (i_iter, CONF.max_ga_iter_steps, float(mmean_length)))


def train(CONF):
    model = create_model(CONF)
    opt = torch.optim.Adam(model.parameters(), CONF.LR)

    s_time = time.time()
    for i_epoch in range(CONF.n_epochs):
        time_0 = time.time()

        n_nodes = [25, 50][i_epoch % 2]
        print("[Epoch: %d/%d]-> nodes = %d | max_ga_iter_steps = %d" % (
            i_epoch + 1, CONF.n_epochs, n_nodes, CONF.max_ga_iter_steps))

        ##############################################################################
        # generate problems
        env_batch = []
        for i_batch in range(CONF.batch_size):
            coords, disMat = gen_random_tsp(n_nodes)  # tsp problem: coords, disMat
            env = TSP_Env(n_nodes, coords, disMat, CONF.pop_size)
            env_batch.append(env)

        ###############################################################################
        # Begin iteration
        for i_iter in range(CONF.max_ga_iter_steps):

            mean_length = np.mean([env.mean_length for env in env_batch])
            print("================= before mean cost: %f =================" % mean_length)

            # Select chromosomes from population
            selected_idxs_batch = []
            for i_batch in range(CONF.batch_size):
                selected_idxs = perform_select(env_batch[i_batch].fits, CONF.n_select_chroms)
                selected_idxs_batch.append(selected_idxs)
            selected_idxs_batch = np.array(selected_idxs_batch)

            time_1 = time.time()
            all_datas = []
            all_states1 = []
            all_states2 = []
            for idx in range(0, CONF.n_select_chroms, 2):
                chrom1_idx_batch = selected_idxs_batch[:, idx]
                chrom1_idx_batch = np.array(chrom1_idx_batch)
                chrom1_nodes_batch = [env.chroms[idx].get_states()[0] for env in env_batch]
                chrom1_edges_batch = [env.chroms[idx].get_states()[1] for env in env_batch]
                chrom1_nodes_batch = np.array(chrom1_nodes_batch)
                chrom1_edges_batch = np.array(chrom1_edges_batch)

                chrom2_idx_batch = selected_idxs_batch[:, idx + 1]
                chrom2_idx_batch = np.array(chrom2_idx_batch)
                chrom2_nodes_batch = [env.chroms[idx + 1].get_states()[0] for env in env_batch]
                chrom2_edges_batch = [env.chroms[idx + 1].get_states()[1] for env in env_batch]
                chrom2_nodes_batch = np.array(chrom2_nodes_batch)
                chrom2_edges_batch = np.array(chrom2_edges_batch)

                edge_index_batch = [np.array(env.edges) for env in env_batch]
                edge_index_batch = np.array(edge_index_batch)

                for i_rollout in range(CONF.n_rollout):
                    # is_last = (i_rollout == CONF.n_rollout - 1)
                    datas, states1, states2 = roll_out(CONF,
                                                       model,
                                                       env_batch,
                                                       edge_index_batch,
                                                       chrom1_idx_batch,
                                                       chrom2_idx_batch,
                                                       chrom1_nodes_batch,
                                                       chrom1_edges_batch,
                                                       chrom2_nodes_batch,
                                                       chrom2_edges_batch,
                                                       CONF.roll_out_steps,
                                                       0.1,
                                                       is_last=False,
                                                       greedy=False)
                    all_datas.extend(datas)
                    all_states1.extend(states1)
                    all_states2.extend(states2)

            time_2 = time.time()
            print("Iter: %d/%d | Rollout finished |Rollout time: %ds" % (i_iter, CONF.max_ga_iter_steps, time_2 - time_1))

            dl = DataLoader(all_datas, batch_size=len(env_batch), shuffle=True)
            for j in range(CONF.train_steps):
                train_once(CONF, model, opt, dl, i_epoch, 0)

            time_3 = time.time()
            print("Iter: %d/%d finished | Train time: %ds" % (i_iter, CONF.max_ga_iter_steps, time_3 - time_2))

        time_4 = time.time()
        print("Epoch: %d/%d finished | Time: %.2fmin" % (i_epoch, CONF.n_epochs, (time_4-time_0)/60.0))

        print("已训练： %.2fmin" % ((time.time() - s_time) / 60.0))

        eval_random(CONF, n_nodes)
        torch.save(model.state_dict(), "pretrained-model/%s.model" % i_epoch)


if __name__ == '__main__':
    args = args()
    train(args)
