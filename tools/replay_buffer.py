import torch
import numpy as np
from torch_geometric.data import Data, DataLoader


class ReplayBuffer(object):
    def __init__(self):
        super(ReplayBuffer, self).__init__()
        self.buf_nodes = []
        self.buf_edges = []
        self.buf_actions = []
        self.buf_rewards = []
        self.buf_values = []
        self.buf_log_probs = []

    def obs(self, nodes, edges, actions, rewards, values, log_probs):
        self.buf_nodes.append(nodes)
        self.buf_edges.append(edges)
        self.buf_actions.append(actions)
        self.buf_rewards.append(rewards)
        self.buf_values.append(values)
        self.buf_log_probs.append(log_probs)

    def compute_values(self, last_v=0, _lambda=1.0):
        rewards = np.array(self.buf_rewards)
        pred_vs = np.array(self.buf_values)

        target_vs = np.zeros_like(rewards)
        advs = np.zeros_like(rewards)

        v = last_v
        for i in reversed(range(rewards.shape[0])):
            v = rewards[i] + _lambda * v
            target_vs[i] = v
            adv = v - pred_vs[i]
            advs[i] = adv

        return target_vs, advs

    def gen_datas(self, last_v, edges_index_batch, batch_size, _lambda=1.0):
        target_vs, advs = self.compute_values(last_v, _lambda)
        advs = (advs - advs.mean()) / advs.std()
        l, w = target_vs.shape

        datas = []
        for i in range(l):
            for j in range(w):
                nodes = self.buf_nodes[i][j]
                edges = self.buf_edges[i][j]
                action = self.buf_actions[i][j]
                edge_index = edges_index_batch[j]
                v = target_vs[i][j]
                adv = advs[i][j]
                log_prob = self.buf_log_probs[i][j]
                #                     print (nodes.dtype,self.edge_index.dtype,edges.dtype,q,action)
                data = Data(x=torch.from_numpy(nodes).float(),
                            edge_index=torch.from_numpy(edge_index).long().T,
                            edge_attr=torch.from_numpy(edges).float(),
                            v=torch.tensor([v]).float(),
                            action=torch.tensor(action).long(),
                            log_prob=torch.tensor([log_prob]).float(),
                            adv=torch.tensor([adv]).float())
                datas.append(data)

        return datas

    def create_data(self, _nodes, _edges_index,  _edges,):
        datas = []
        batch_size = len(_nodes)
        for i in range(batch_size):
            nodes = _nodes[i]
            edges = _edges[i]
            edges_index = _edges_index[i]
            data = Data(x=torch.from_numpy(nodes).float(),
                        edge_index=torch.from_numpy(edges_index).long().T,
                        edge_attr=torch.from_numpy(edges).float())
            datas.append(data)
        dl = DataLoader(datas, batch_size=batch_size)
        return list(dl)[0]
