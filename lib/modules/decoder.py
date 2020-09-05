import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from lib.modules.attention import Attention


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cell = torch.nn.GRUCell(input_dim, hidden_dim, bias=True)
        self.attn = Attention(hidden_dim)

    def evaluate(self, encoder_inputs, hx, actions):
        _input = encoder_inputs.new_zeros((encoder_inputs.size(0), encoder_inputs.size(2)))
        mask = encoder_inputs.new_zeros((encoder_inputs.size(0), encoder_inputs.size(1)))
        log_ps = []
        entropys = []

        actions = actions.transpose(0, 1)
        for act in actions:
            hx = self.cell(_input, hx)
            p = self.attn(hx, encoder_inputs, mask)
            dist = Categorical(p)
            entropy = dist.entropy()

            log_p = dist.log_prob(act)
            log_ps.append(log_p)
            mask = mask.scatter(1, act.unsqueeze(-1).expand(mask.size(0), -1), 1)
            _input = torch.gather(encoder_inputs, 1, act.unsqueeze(-1).unsqueeze(-1).expand(encoder_inputs.size(0), -1,
                                                                                            encoder_inputs.size(
                                                                                                2))).squeeze(1)
            entropys.append(entropy)

        log_ps = torch.stack(log_ps, 1)
        entropys = torch.stack(entropys, 1)
        log_p = log_ps.sum(dim=1)
        entropy = entropys.mean(dim=1)

        return log_p, entropy

    def forward(self, encoder_inputs, hx, n_steps, greedy=False):
        _input = encoder_inputs.new_zeros((encoder_inputs.size(0), encoder_inputs.size(2)))
        mask = encoder_inputs.new_zeros((encoder_inputs.size(0), encoder_inputs.size(1)))
        log_ps = []
        actions = []
        entropys = []

        for i in range(n_steps):
            hx = self.cell(_input, hx)
            p = self.attn(hx, encoder_inputs, mask)
            dist = Categorical(p)
            entropy = dist.entropy()

            if greedy:
                _, index = p.max(dim=-1)
            else:
                index = dist.sample()
            log_p = dist.log_prob(index)

            actions.append(index)
            log_ps.append(log_p)
            entropys.append(entropy)

            mask = mask.scatter(1, index.unsqueeze(-1).expand(mask.size(0), -1), 1)
            _input = torch.gather(encoder_inputs, 1,
                                  index.unsqueeze(-1).unsqueeze(-1).expand(encoder_inputs.size(0), -1,
                                                                           encoder_inputs.size(2))).squeeze(1)
        log_ps = torch.stack(log_ps, 1)
        actions = torch.stack(actions, 1)
        entropys = torch.stack(entropys, 1)
        log_p = log_ps.sum(dim=1)
        entropy = entropys.mean(dim=1)

        return actions, log_p, entropy
