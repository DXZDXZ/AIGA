import torch.nn as nn

from lib.modules import encoder, decoder


class Critic(nn.Module):
    def __init__(self, hidden_node_dim):
        super(Critic, self).__init__()
        self.v1 = nn.Linear(hidden_node_dim, hidden_node_dim)
        self.v2 = nn.Linear(hidden_node_dim, 1)

    def forward(self, datas):
        out = self.v1(datas)
        out = self.v2(out)
        return out


class TSP_Model(nn.Module):

    def __init__(self, CONF):
        super(TSP_Model, self).__init__()
        self.encoder = encoder.Encoder(CONF.input_node_dim, CONF.hidden_node_dim, CONF.input_edge_dim, CONF.hidden_edge_dim, CONF.conv_layers)
        self.decoder = decoder.Decoder(CONF.hidden_node_dim, CONF.hidden_node_dim)
        self.critic = Critic(CONF.hidden_node_dim)

    def evaluate(self, datas, actions):
        x = self.encoder(datas)
        pooled = x.mean(dim=1)
        log_p, entropy = self.decoder.evaluate(x, pooled, actions)

        values = self.critic(pooled)
        values = values.squeeze(-1)

        return log_p, values, entropy

    def forward(self, datas, steps, greedy=False):
        x = self.encoder(datas)
        pooled = x.mean(dim=1)

        actions, log_p, entropy = self.decoder(x, pooled, steps, greedy)

        values = self.critic(pooled)
        values = values.squeeze(-1)

        return actions, log_p, values, entropy
