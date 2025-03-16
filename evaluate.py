# %%
import json
import torch
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from models.dkt_plus import DKTPlus
from models.dkt import DKT
from models.dkvmn import DKVMN
from models.sakt import SAKT

# model_name = "dkt"
# model_name = "dkt+"
model_name = "dkvmn"
# model_name = "sakt"
dataset_name = "ASSIST2015"
len_th = 110
# %%
def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)
def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)
def mv_avg(x, window=11):
    return np.convolve(x, np.ones(window), 'valid') / window
def dkvmn_all(model, u_q_seqs, u_r_seqs):
    self = model
    q = u_q_seqs
    r = u_r_seqs
    x = q + self.num_q * r

    batch_size = x.shape[0]
    Mvt = self.Mv0.unsqueeze(0).repeat(batch_size, 1, 1)

    Mv = [Mvt]

    k = self.k_emb_layer(q)
    v = self.v_emb_layer(x)

    w = torch.softmax(torch.matmul(k, self.Mk.T), dim=-1)

    # Write Process
    e = torch.sigmoid(self.e_layer(v))
    a = torch.tanh(self.a_layer(v))

    for et, at, wt in zip(
        e.permute(1, 0, 2), a.permute(1, 0, 2), w.permute(1, 0, 2)
    ):
        Mvt = Mvt * (1 - (wt.unsqueeze(-1) * et.unsqueeze(1))) + \
            (wt.unsqueeze(-1) * at.unsqueeze(1))
        Mv.append(Mvt)

    Mv = torch.stack(Mv, dim=1)

    all_p = []
    for qi in range(num_q):
        q2 = torch.zeros(u_q_seqs.shape[1]).type(torch.LongTensor).to(device) + qi
        k2 = self.k_emb_layer(q2)
        w2 = torch.softmax(torch.matmul(k2, self.Mk.T), dim=-1)
        # Read Process
        f = torch.tanh(
            self.f_layer(
                torch.cat(
                    [
                        (w2.unsqueeze(-1) * Mv[:, :-1]).sum(-2),
                        k
                    ],
                    dim=-1
                )
            )
        )
        p = torch.sigmoid(self.p_layer(f)).squeeze()
        all_p.append(p)
    all_p = torch.stack(all_p, dim=-1)
    return all_p
# %%
#### Load datas

ckpts_dir_path = f"ckpts/{model_name}/{dataset_name}"
model_config = load_json(f"{ckpts_dir_path}/model_config.json")
train_config = load_json(f"{ckpts_dir_path}/train_config.json")
ckpt_path = f"{ckpts_dir_path}/model.ckpt"

dataset_dir_path = f"datasets/{dataset_name}"
q_list = load_pickle(f"{dataset_dir_path}/q_list.pkl")
num_q = len(q_list)
q_seqs = load_pickle(f"{dataset_dir_path}/q_seqs.pkl")
r_seqs = load_pickle(f"{dataset_dir_path}/r_seqs.pkl")
train_indeces = load_pickle(f"{dataset_dir_path}/train_indices.pkl")
test_indeces = load_pickle(f"{dataset_dir_path}/test_indices.pkl")
org_indexes = load_pickle(f"{dataset_dir_path}/org_idxs.pkl")

# testのindexを取得
test_org_indeces = [org_indexes[i] for i in test_indeces]
# qとrをtestのindexのみにする
q_seqs = [q_seqs[i] for i in test_org_indeces]
r_seqs = [r_seqs[i] for i in test_org_indeces]
# %%
#### Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
if model_name == "dkt+":
    model = DKTPlus(num_q, **model_config).to(device)
if model_name == "dkt":
    model = DKT(num_q, **model_config).to(device)
if model_name == "dkvmn":
    model = DKVMN(num_q, **model_config).to(device)
if model_name == "sakt":
    model = SAKT(num_q, **model_config).to(device)

model.load_state_dict(torch.load(ckpt_path))
model = model.eval()
# %%
#### check user's sequence length
cnt_df = pd.DataFrame(
    [{"user_idx": ui, "size": len(q_seq)}
     for ui, q_seq in enumerate(q_seqs)]
)
size_counts = (cnt_df['size']
               .value_counts()
               .sort_index(ascending=False)
               .cumsum()[::-1]) 
user_size = size_counts[len_th]
print(f"Sequence Length <= {len_th}: {user_size}")
plt.figure(figsize=(5, 3))
plt.title(f"{model_name} - {dataset_name},\n Sequence Length <= {len_th}: {user_size}")
plt.plot(size_counts.index, size_counts.values)
plt.xlabel("Sequence Length")
plt.ylabel("user size of x<=Sequence Length")
plt.axvline(x=len_th, color='r', linestyle='--')
plt.axhline(y=user_size, color='r', linestyle='--')
plt.pause(0.1)
plt.close()
# %%
#### predict user's correct rate
all_ys_btc = []
all_rs_bt = []
all_qs_bt = []
for ui in range(len(q_seqs)):
    if len(q_seqs[ui]) == 1:
        continue
    if len(q_seqs[ui]) < len_th:
        continue
    u_q_seqs = torch.tensor([q_seqs[ui]]).type(torch.LongTensor).to(device)
    u_r_seqs = torch.tensor([r_seqs[ui]]).type(torch.LongTensor).to(device)
    u_q_seqs = u_q_seqs[:, :len_th]
    u_r_seqs = u_r_seqs[:, :len_th]
    q = u_q_seqs[:, :-1]
    r = u_r_seqs[:, :-1]
    qshft = u_q_seqs[:, 1:]
    rshft = u_r_seqs[:, 1:]

    if model_name == "dkt" or model_name == "dkt+":
        y = model(q, r)
        use_r = rshft
        use_q = qshft
        y = y[0].detach().cpu().numpy()
    if model_name == "dkvmn":
        y = dkvmn_all(model, q, r)
        y = y.detach().cpu().numpy()
        use_r = r
        use_q = q

    use_r = use_r[0].detach().cpu().numpy()
    use_q = use_q[0].detach().cpu().numpy()

    all_ys_btc.append(y)
    all_rs_bt.append(use_r)
    all_qs_bt.append(use_q)
all_ys_btc = np.stack(all_ys_btc)
all_rs_bt = np.stack(all_rs_bt)
all_qs_bt = np.stack(all_qs_bt)

# %%
#### plot user's correct rate
res_tc = all_ys_btc.mean(axis=0)
res_t_std = res_tc.std(axis=1)
res_t = res_tc.mean(axis=1)
R = 3
plt.figure(figsize=(5, R*1.7))
plt.suptitle(f"{model_name} - {dataset_name}")
plt.subplots_adjust(hspace=0.8)
plt.subplot(R, 1, 1)
plt.title("Correct Rate(auto scale)")
plt.plot(res_t)
plt.plot(mv_avg(res_t), alpha=0.5)
plt.xlabel("Time")
plt.ylabel("Mean Correct Rate")
plt.subplot(R, 1, 2)
plt.title("Correct Rate(fix scale)")
plt.plot(res_t)
plt.plot(mv_avg(res_t), alpha=0.5)
plt.ylabel("Mean Correct Rate")
plt.ylim(0, 1)
plt.subplot(R, 1, 3)
plt.title("Correct Rate std(auto scale)")
plt.plot(res_t_std)
plt.plot(mv_avg(res_t_std), alpha=0.5)
plt.ylabel("Std Correct Rate")
plt.pause(0.1)
plt.close()
# %%
#### auc
l_res_bt = []
l_r_seqs_bt = all_rs_bt
for bi in range(all_ys_btc.shape[0]):
    idxs = all_qs_bt[bi, :]
    l_res_bt.append(all_ys_btc[bi, np.arange(len(idxs)), idxs])
l_res_bt = np.stack(l_res_bt)

l_res_bt.shape
l_r_seqs_bt.shape

auc_scores = []
for t in range(l_res_bt.shape[1]):
    preds = l_res_bt[:, t]
    trues = l_r_seqs_bt[:, t]
    score = roc_auc_score(y_true=trues, y_score=preds)
    auc_scores.append(score)
auc_scores = np.array(auc_scores)
all_score = roc_auc_score(y_true=l_r_seqs_bt.reshape(-1), y_score=l_res_bt.reshape(-1))

R = 2
plt.figure(figsize=(5, R*1.8))
plt.suptitle(f"{model_name} - {dataset_name}, all_score: {all_score:.3f}")
plt.subplots_adjust(hspace=0.8)
plt.subplot(R, 1, 1)
plt.title("AUC Score(auto scale)")
plt.plot(auc_scores)
plt.plot(mv_avg(auc_scores), alpha=0.5)
plt.xlabel("Time")
plt.ylabel("AUC Score")

plt.subplot(R, 1, 2)
plt.title("AUC Score(fix scale)")
plt.plot(auc_scores)
plt.plot(mv_avg(auc_scores), alpha=0.5)
plt.ylabel("AUC Score")
plt.ylim(0, 1)
plt.pause(0.1)

