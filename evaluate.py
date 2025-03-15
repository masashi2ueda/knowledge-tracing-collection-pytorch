# %%
import json
import torch
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from models.dkt_plus import DKTPlus
from models.dkt import DKT
from models.dkvmn import DKVMN
from models.sakt import SAKT

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
        q2 = torch.zeros(len_th).type(torch.LongTensor).to(device) + qi
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
    all_p = torch.stack(all_p, dim=2)
    return all_p
# %%
#### Load datas
# model_name = "dkt+"
# model_name = "dkt"
model_name = "dkvmn"
# model_name = "sakt"
dataset_name = "ASSIST2015"
ckpts_dir_path = f"ckpts/{model_name}/{dataset_name}"
model_config = load_json(f"{ckpts_dir_path}/model_config.json")
train_config = load_json(f"{ckpts_dir_path}/train_config.json")
ckpt_path = f"{ckpts_dir_path}/model.ckpt"

dataset_dir_path = f"datasets/{dataset_name}"
q_list = load_pickle(f"{dataset_dir_path}/q_list.pkl")
num_q = len(q_list)
q_seqs = load_pickle(f"{dataset_dir_path}/q_seqs.pkl")
r_seqs = load_pickle(f"{dataset_dir_path}/r_seqs.pkl")

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
model.eval

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

len_th = 110
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
user_idxs = cnt_df[len_th <= cnt_df["size"]]["user_idx"].values
user_idxs
u_q_seqs = [q_seqs[ui][:len_th] for ui in user_idxs]
u_r_seqs = [r_seqs[ui][:len_th] for ui in user_idxs]
u_q_seqs, u_r_seqs = np.stack(u_q_seqs), np.stack(u_r_seqs)
def to_tensor(x):
    return torch.tensor(x).type(torch.LongTensor).to(device)
u_q_seqs, u_r_seqs = to_tensor(u_q_seqs), to_tensor(u_r_seqs)
if model_name == "dkt" or model_name == "dkt+":
    res_btc = model(u_q_seqs, u_r_seqs)
if model_name == "dkvmn":
    res_btc = dkvmn_all(model, u_q_seqs, u_r_seqs)
# %%
#### plot user's correct rate
res_tc = res_btc.mean(dim=0)
res_t_std = res_tc.std(dim=1)
res_t = res_tc.mean(dim=1)
res_t = res_t.cpu().detach().numpy()
res_t_std = res_t_std.cpu().detach().numpy()
plt.figure(figsize=(5, 5))
plt.suptitle(f"{model_name} - {dataset_name}")
plt.subplots_adjust(hspace=0.5)
plt.subplot(2, 1, 1)
plt.plot(res_t)
plt.plot(mv_avg(res_t), alpha=0.5)
plt.xlabel("Time")
plt.ylabel("Mean Correct Rate")
plt.subplot(2, 1, 2)
plt.plot(res_t_std)
plt.plot(mv_avg(res_t_std), alpha=0.5)
plt.ylabel("Std Correct Rate")
plt.pause(0.1)
plt.close()
# %%
#### plot error status
if model_name == "dkt" or model_name == "dkt+":
    l_q_seqs_bt = u_q_seqs[1:]
    l_r_seqs_bt = u_r_seqs[1:]
    l_res_btc = res_btc[:1]
if model_name == "dkvmn":
    l_q_seqs_bt = u_q_seqs
    l_r_seqs_bt = u_r_seqs
    l_res_btc = res_btc

l_q_seqs_bt = l_q_seqs_bt.cpu().detach().numpy()
l_r_seqs_bt = l_r_seqs_bt.cpu().detach().numpy()
l_res_btc = l_res_btc.cpu().detach().numpy()

batch_size, time_steps, _ = l_res_btc.shape
batch_idx = torch.arange(batch_size).unsqueeze(1)  # (batch_size, 1)
time_idx = torch.arange(time_steps).unsqueeze(0)   # (1, time_steps)
l_res_bt = l_res_btc[batch_idx, time_idx, l_q_seqs_bt]

diff_bt = np.abs(l_res_bt - l_r_seqs_bt)
diff_t = diff_bt.mean(axis=0)
diff_std_t = diff_bt.std(axis=0)
plt.figure(figsize=(5, 5))
plt.suptitle(f"{model_name} - {dataset_name}")
plt.subplots_adjust(hspace=0.5)
plt.subplot(2, 1, 1)
plt.plot(diff_t)
plt.plot(mv_avg(diff_t), alpha=0.5)
plt.xlabel("Time")
plt.ylabel("Mean Absolute Error")
plt.subplot(2, 1, 2)
plt.plot(diff_std_t)
plt.plot(mv_avg(diff_std_t), alpha=0.5)
# plt.xlabel("Time")
plt.ylabel("Std Absolute Error")
plt.pause(0.1)
plt.close()
# %%
