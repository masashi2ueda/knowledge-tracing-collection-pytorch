import os

import numpy as np
import torch

from torch.nn import Module, Embedding, LSTM, Linear, Dropout
from torch.nn.functional import one_hot, binary_cross_entropy
from sklearn import metrics


class DKTPlus(Module):
    def __init__(
        self, num_q, emb_size, hidden_size, lambda_r, lambda_w1, lambda_w2
    ):
        super().__init__()
        self.num_q = num_q
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.lambda_r = lambda_r
        self.lambda_w1 = lambda_w1
        self.lambda_w2 = lambda_w2

        self.interaction_emb = Embedding(self.num_q * 2, self.emb_size)
        self.lstm_layer = LSTM(
            self.emb_size, self.hidden_size, batch_first=True
        )
        self.out_layer = Linear(self.hidden_size, self.num_q)
        self.dropout_layer = Dropout()

    def forward(self, q, r):
        x = q + self.num_q * r

        h, _ = self.lstm_layer(self.interaction_emb(x))
        y = self.out_layer(h)
        y = self.dropout_layer(y)
        y = torch.sigmoid(y)

        return y

    def train_model(
        self, train_loader, test_loader, num_epochs, opt, ckpt_path
    ):
        aucs = []
        loss_means = []

        for i in range(1, num_epochs + 1):
            loss_mean = []

            for data in train_loader:
                q, r, qshft, rshft, m = data

                self.train()

                y = self(q.long(), r.long())
                y_curr = (y * one_hot(q.long(), self.num_q)).sum(-1)
                y_next = (y * one_hot(qshft.long(), self.num_q)).sum(-1)

                y_curr = torch.masked_select(y_curr, m)
                y_next = torch.masked_select(y_next, m)
                r = torch.masked_select(r, m)
                rshft = torch.masked_select(rshft, m)

                opt.zero_grad()
                loss = \
                    binary_cross_entropy(y_next, rshft) + \
                    self.lambda_r * binary_cross_entropy(y_curr, r) + \
                    self.lambda_w1 * \
                    torch.norm(y[:, 1:] - y[:, :-1], p=1, dim=-1).mean() / \
                    self.num_q + \
                    self.lambda_w2 * \
                    (torch.norm(y[:, 1:] - y[:, :-1], p=2, dim=-1) ** 2)\
                    .mean() / \
                    self.num_q
                loss.backward()
                opt.step()

                loss_mean.append(loss.detach().cpu().numpy())

            with torch.no_grad():
                for data in test_loader:
                    q, r, qshft, rshft, m = data

                    self.eval()

                    y = self(q.long(), r.long())
                    y_next = (y * one_hot(qshft.long(), self.num_q)).sum(-1)

                    y_next = torch.masked_select(y_next, m).detach().cpu()
                    rshft = torch.masked_select(rshft, m).detach().cpu()

                    auc = metrics.roc_auc_score(
                        y_true=rshft.numpy(), y_score=y_next.numpy()
                    )

                    loss_mean = np.mean(loss_mean)

                    print(
                        "Epoch: {},   AUC: {},   Loss Mean: {}"
                        .format(i, auc, loss_mean)
                    )

                    if i == 1 or auc > aucs[-1]:
                        torch.save(
                            self.state_dict(),
                            os.path.join(
                                ckpt_path, "model.ckpt"
                            )
                        )

                    aucs.append(auc)
                    loss_means.append(loss_mean)

        return aucs, loss_means
