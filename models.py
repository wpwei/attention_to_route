import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.distributions.categorical import Categorical
from datasets import TSPTrainSet, TSPTestSet
from torch.utils.data import DataLoader
from problems import TSP
from utils import load_dataset
from scipy import stats
import numpy as np


class Normalization(nn.Module):
    """
    1D batch normalization for [*, C] input
    """
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.norm = nn.BatchNorm1d(feature_dim)

    def forward(self, x):
        size = x.size()
        return self.norm(x.view(-1, size[-1])).view(*size)


class Attention(nn.Module):
    """
    Multi-head attention

    Input:
      q: [batch_size, n_node, hidden_dim]
      k, v: q if None

    Output:
      att: [n_node, hidden_dim]
    """
    def __init__(self,
                 q_hidden_dim,
                 k_dim,
                 v_dim,
                 n_head,
                 k_hidden_dim=None,
                 v_hidden_dim=None):
        super().__init__()
        self.q_hidden_dim = q_hidden_dim
        self.k_hidden_dim = k_hidden_dim if k_hidden_dim else q_hidden_dim
        self.v_hidden_dim = v_hidden_dim if v_hidden_dim else q_hidden_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.n_head = n_head

        self.proj_q = nn.Linear(q_hidden_dim, k_dim * n_head, bias=False)
        self.proj_k = nn.Linear(self.k_hidden_dim, k_dim * n_head, bias=False)
        self.proj_v = nn.Linear(self.v_hidden_dim, v_dim * n_head, bias=False)
        self.proj_output = nn.Linear(v_dim * n_head,
                                     self.v_hidden_dim,
                                     bias=False)

    def forward(self, q, k=None, v=None, mask=None):
        if k is None:
            k = q
        if v is None:
            v = k
        if v is None:
            v = q

        bsz, n_node, hidden_dim = q.size()

        qs = torch.stack(torch.chunk(self.proj_q(q), self.n_head, dim=-1),
                         dim=1)  # [batch_size, n_head, n_node, k_dim]
        ks = torch.stack(torch.chunk(self.proj_k(k), self.n_head, dim=-1),
                         dim=1)  # [batch_size, n_head, n_node, k_dim]
        vs = torch.stack(torch.chunk(self.proj_v(v), self.n_head, dim=-1),
                         dim=1)  # [batch_size, n_head, n_node, v_dim]

        normalizer = self.k_dim**0.5
        u = torch.matmul(qs, ks.transpose(2, 3)) / normalizer
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            u = u.masked_fill(mask, float('-inf'))
        att = torch.matmul(torch.softmax(u, dim=-1), vs)
        att = att.transpose(1, 2).reshape(bsz, n_node,
                                          self.v_dim * self.n_head)
        att = self.proj_output(att)
        return att


class TSPEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 ff_dim,
                 n_layer,
                 k_dim=16,
                 v_dim=16,
                 n_head=8):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ff_dim = ff_dim
        self.n_layer = n_layer
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.n_head = n_head

        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.attentions = nn.ModuleList([
            Attention(hidden_dim, k_dim, v_dim, n_head) for _ in range(n_layer)
        ])
        self.ff = nn.ModuleList([
            nn.Sequential(*[
                nn.Linear(hidden_dim, ff_dim),
                nn.ReLU(),
                nn.Linear(ff_dim, hidden_dim)
            ]) for _ in range(n_layer)
        ])
        self.bn1 = nn.ModuleList(
            [Normalization(hidden_dim) for _ in range(n_layer)])
        self.bn2 = nn.ModuleList(
            [Normalization(hidden_dim) for _ in range(n_layer)])

    def forward(self, x):
        h = self.embedding(x)
        for i in range(self.n_layer):
            h = self.bn1[i](h + self.attentions[i](h))
            h = self.bn2[i](h + self.ff[i](h))
        return h


class TSPDecoder(nn.Module):
    def __init__(self, hidden_dim, k_dim, v_dim, n_head):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.n_head = n_head

        self.v_l = nn.Parameter(
            torch.FloatTensor(size=[1, 1, hidden_dim]).uniform_())
        self.v_f = nn.Parameter(
            torch.FloatTensor(size=[1, 1, hidden_dim]).uniform_())

        self.attention = Attention(hidden_dim * 3,
                                   k_dim,
                                   v_dim,
                                   n_head,
                                   k_hidden_dim=hidden_dim,
                                   v_hidden_dim=hidden_dim)

        self.proj_k = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x, C=10, rollout=False):
        """
        x: [batch_size, n_node, hidden_dim] node embeddings of TSP graph

        """
        bsz, n_node, hidden_dim = x.size()

        h_avg = x.mean(dim=-2, keepdim=True)
        first = self.v_f.repeat(bsz, 1, 1)
        last = self.v_l.repeat(bsz, 1, 1)

        k = self.proj_k(x)

        normalizer = self.hidden_dim**0.5
        visited_idx = []
        mask = torch.zeros([bsz, n_node], device=x.device).bool()
        log_prob = 0

        for i in range(n_node):
            h_c = torch.cat([h_avg, last, first], -1)
            q = self.attention(h_c, x, x, mask=mask)

            u = torch.tanh(q.bmm(k.transpose(-2, -1)) / normalizer) * C
            u = u.masked_fill(mask.unsqueeze(1), float('-inf'))

            if rollout:
                visit_idx = u.max(-1)[1]
            else:
                m = Categorical(logits=u)
                visit_idx = m.sample()
                log_prob += m.log_prob(visit_idx)

            visited_idx += [visit_idx]
            mask = mask.scatter(1, visit_idx, True)

            visit_idx = visit_idx.unsqueeze(-1).repeat(1, 1, hidden_dim)
            last = torch.gather(x, 1, visit_idx)
            if len(visited_idx) == 1:
                first = last

        visited_idx = torch.cat(visited_idx, -1)
        return visited_idx, log_prob


class TSPSolver(nn.Module):
    def __init__(self,
                 input_dim=2,
                 hidden_dim=128,
                 ff_dim=512,
                 n_layer=3,
                 k_dim=16,
                 v_dim=16,
                 n_head=8):
        super().__init__()
        self.encoder = TSPEncoder(input_dim, hidden_dim, ff_dim, n_layer,
                                  k_dim, v_dim, n_head)
        self.decoder = TSPDecoder(hidden_dim, k_dim, v_dim, n_head)

    def forward(self, x, rollout=False):
        x = self.encoder(x)
        return self.decoder(x, rollout=rollout)


class TSPAgent(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.net = TSPSolver(self.hparams.input_dim, self.hparams.hidden_dim,
                             self.hparams.ff_dim, self.hparams.n_layer,
                             self.hparams.k_dim, self.hparams.v_dim,
                             self.hparams.n_head)
        self.M = None
        self.beta = 0.8
        self.target_net = TSPSolver(self.hparams.input_dim,
                                    self.hparams.hidden_dim,
                                    self.hparams.ff_dim, self.hparams.n_layer,
                                    self.hparams.k_dim, self.hparams.v_dim,
                                    self.hparams.n_head)
        self.target_net.load_state_dict(self.net.state_dict())

    def forward(self, x, rollout=False):
        return self.net(x, rollout=rollout)

    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=self.hparams.lr)

    def train_dataloader(self):
        train_set = TSPTrainSet(self.hparams.n_batch_per_epoch,
                                self.hparams.batch_size, self.hparams.n_node)

        # Also setup initial baseline dataset and dataloader
        self._bdl = self.baseline_dataloader()

        return DataLoader(train_set, batch_size=None, num_workers=1)

    def baseline_dataloader(self):
        baseline_set = TSPTestSet(
            np.random.uniform(
                size=[self.hparams.baseline_set_size, self.hparams.n_node, 2
                      ]).astype(np.float32))
        return DataLoader(baseline_set,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers)

    def training_step(self, batch, batch_idx):
        self.net = self.net.train()
        perm, log_prob = self.net(batch)

        with torch.no_grad():
            cost = TSP(batch, perm)

            # rollout baseline
            self.target_net = self.target_net.eval()
            baseline_perm, _ = self.target_net(batch, rollout=True)
            baseline_cost = TSP(batch, baseline_perm)
            advatange = cost - baseline_cost

            # exponential baseline
            # if self.M is None:
            #     self.M = cost.mean()
            # else:
            #     self.M = self.beta * self.M + (1. - self.beta) * cost.mean()
            # advatange = cost - self.M

        loss = advatange * log_prob
        loss = loss.mean()
        logs = {'adv': advatange.mean().item(), 'loss': loss.item()}

        return {'loss': loss, 'log': logs}

    def on_epoch_end(self):
        """
        Paired t-test on baseline set.
        Update target net if improvement is significant (p <= 0.05).
        Resample baseline test if target net updated.
        """
        with torch.no_grad():
            self.target_net = self.target_net.eval()
            self.net = self.net.eval()

            baseline_cost = []
            policy_cost = []

            for batch in iter(self._bdl):
                if self.on_gpu:
                    batch = batch.cuda()
                baseline_perm, _ = self.target_net(batch, rollout=True)
                baseline_c = TSP(batch, baseline_perm)
                baseline_cost += [baseline_c.cpu().numpy()]

                policy_perm, _ = self.net(batch, rollout=True)
                policy_c = TSP(batch, policy_perm)
                policy_cost += [policy_c.cpu().numpy()]

            self.net = self.net.train()

            baseline_cost = np.concatenate(baseline_cost).reshape(-1)
            policy_cost = np.concatenate(policy_cost).reshape(-1)

            improve = (baseline_cost - policy_cost).mean() >= 0
            _, p_value = stats.ttest_rel(baseline_cost, policy_cost)
            if improve and p_value <= 0.05:
                self.target_net.load_state_dict(self.net.state_dict())
                self._bdl = self.baseline_dataloader()

    def val_dataloader(self):
        val_data = load_dataset(
            f'data/tsp/tsp{self.hparams.n_node}_validation_seed4321.pkl')
        val_set = TSPTestSet(val_data)
        return DataLoader(val_set,
                          batch_size=self.hparams.batch_size,
                          num_workers=4)

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            perm, _ = self.net(batch, rollout=True)
            cost = TSP(batch, perm)

            return {'val_cost': cost}

    def validation_epoch_end(self, outputs):
        val_cost = torch.cat([x['val_cost'] for x in outputs]).mean()
        gap = (val_cost - self.hparams.val_optimal) / self.hparams.val_optimal
        return {'val_loss': gap, 'log': {'val_cost': val_cost, 'val_gap': gap}}
