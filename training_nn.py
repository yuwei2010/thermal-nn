import optuna
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Parameter as TorchParam
from torch import Tensor
from typing import List, Tuple
from joblib import parallel_backend
from torch.jit import ScriptModule, script_method

#%%

MeasurementData_Merged = pd.read_csv(r"../../MeasurementData_Merged_EFAD.csv", index_col=0)
MeasurementData_Merged['pm'] = MeasurementData_Merged[['ExternTemp.rp_rbe_Cif_10ms_PIExternTemp1.rbe_Cif', 'ExternTemp.rp_rbe_Cif_10ms_PIExternTemp10.rbe_Cif', 'ExternTemp.rp_rbe_Cif_10ms_PIExternTemp5.rbe_Cif','ExternTemp.rp_rbe_Cif_10ms_PIExternTemp7.rbe_Cif']].mean(axis=1)
MeasurementData_Merged['stator_winding'] = MeasurementData_Merged[['I_EM_tTMotWinDeLay2Cl03', 'I_EM_tTMotWinDeLay2Cl12','I_EM_tTMotWinDeLay5Cl03','I_EM_tTMotWinDeLay5Cl06','I_EM_tTMotWinDeLay5Cl12',
                                                                   'I_EM_tTMotWinNdeLay2Cl03','I_EM_tTMotWinNdeLay2Cl12','I_EM_tTMotWinNdeLay5Cl03']].mean(axis=1)
MeasurementData_Merged['Us'] = np.sqrt(MeasurementData_Merged['uDaFundaFild100ms.Rec_10ms_Fild.pp_rbe_Mct_10ms_Fild.rbe_MctAsm']**2 + MeasurementData_Merged['uQaFundaFild100ms.Rec_10ms_Fild.pp_rbe_Mct_10ms_Fild.rbe_MctAsm']**2)
MeasurementData_Merged['Is'] = np.sqrt(MeasurementData_Merged['iDaFild10ms.Rec_2ms_Fild.rp_rbe_CddIPha_2ms_Fild.rbe_MctAsm']**2 + MeasurementData_Merged['iQaFild10ms.Rec_2ms_Fild.rp_rbe_CddIPha_2ms_Fild.rbe_MctAsm']**2)
#%%

input_cols = [
                'nEmFild100ms.Rec_10ms_Fild.pp_rbe_CddAgEm_10ms_Fild.rbe_CddRslvr',
                'tqEmFild100ms.Rec_10ms_Fild.pp_rbe_Mct_10ms_Fild.rbe_MctAsm',
                'iDaFild10ms.Rec_2ms_Fild.rp_rbe_CddIPha_2ms_Fild.rbe_MctAsm',
                'iQaFild10ms.Rec_2ms_Fild.rp_rbe_CddIPha_2ms_Fild.rbe_MctAsm',
                'uDaFundaFild100ms.Rec_10ms_Fild.pp_rbe_Mct_10ms_Fild.rbe_MctAsm',
                'uQaFundaFild100ms.Rec_10ms_Fild.pp_rbe_Mct_10ms_Fild.rbe_MctAsm',
                'R_CW_tCooltIvtrOut',
                'tEmSnsrFild10ms.Rec_2ms_Fild.rp_rbe_CddTEm_2ms_Fild.rbe_TMdlEm'
                ]
data = MeasurementData_Merged.copy()
data = data[data['pm'] <= 200]
#%%
target_cols = ['pm','stator_winding','R_EM_tTMotRshaftOilIn']
temperature_cols = target_cols + ['tEmSnsrFild10ms.Rec_2ms_Fild.rp_rbe_CddTEm_2ms_Fild.rbe_TMdlEm','R_CW_tCooltIvtrOut']
#%%
test_profiles = [36, 37]
test_blacklist_profiles = [25]
train_profiles = [p for p in data.profile_id.unique() if p not in test_profiles and p not in test_blacklist_profiles]
profile_sizes = data.groupby("profile_id").agg("size")
#%%
# normalize
non_temperature_cols = [c for c in data if c in input_cols and c not in temperature_cols]
data.loc[:, temperature_cols] /= 200  # deg C
data.loc[:, non_temperature_cols] /= data.loc[:, non_temperature_cols].abs().max(axis=0)

#%%
device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#%%
data = data.loc[:, input_cols + ["profile_id"] + target_cols].dropna()


#%%
def generate_tensor(profiles_list):
    """Returns profiles of the data set in a coherent 3D tensor with
    time-major shape (T, B, F) where
    T : Maximum profile length
    B : Batch size = Amount of profiles
    F : Amount of input features.

    Also returns a likewise-shaped sample_weights tensor, which zeros out post-padded zeros for use
    in the cost function (i.e., it acts as masking tensor)"""
    tensor = np.full(
        (profile_sizes[profiles_list].max(), len(profiles_list), data.shape[1] - 1),
        np.nan,
    )
    for i, (pid, df) in enumerate(
        data.loc[data.profile_id.isin(profiles_list), :].groupby("profile_id")
    ):
        assert pid in profiles_list, f"PID is not in {profiles_list}!"
        tensor[: len(df), i, :] = df.drop(columns="profile_id").to_numpy()
    sample_weights = 1 - np.isnan(tensor[:, :, 0])
    tensor = np.nan_to_num(tensor).astype(np.float32)
    tensor = torch.from_numpy(tensor).to(device)
    sample_weights = torch.from_numpy(sample_weights).to(device)
    return tensor, sample_weights

#%%
train_tensor, train_sample_weights = generate_tensor(train_profiles)
test_tensor, test_sample_weights = generate_tensor(test_profiles)
print(f"Train tensor shape: {train_tensor.shape}, Sample weights shape: {train_sample_weights.shape}")

#%%
p = Path(r'data')
p.mkdir(exist_ok=True, parents=True)
torch.save(train_tensor, p / "train_tensor.pt")
torch.save(train_sample_weights, p / "train_sample_weights.pt")
torch.save(test_tensor, p / "test_tensor.pt")
torch.save(test_sample_weights, p / "test_sample_weights.pt")
#%%
# Hyper parameters optimization 
# 自定义正弦激活层
class SinusLayer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)

def smooth_abs(x: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    return torch.sqrt(x**2 + epsilon)

# 激活函数映射
def get_activation(activation_name: str) -> nn.Module:
    activation_dict = {
        "Sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "linear": nn.Identity(),
        "ReLU": nn.ReLU(),
        "biased Elu": nn.ELU(alpha=1.0),
        "sinus": SinusLayer()
    }
    return activation_dict[activation_name]
#%%
# TNNCell定义（支持动态结构和激活函数）
class TNNCell(nn.Module):
    def __init__(self, cond_net_units, cond_activations, 
                       ploss_net_units, ploss_activations):
        super().__init__()
        self.sample_time = 0.5
        self.output_size = len(target_cols)
        self.caps = nn.Parameter(torch.Tensor(self.output_size))
        nn.init.normal_(self.caps, mean=-9.2, std=0.5)
        
        n_temps = len(temperature_cols) # number of temperatures (targets and input)
        n_conds = int(0.5 * n_temps * (n_temps - 1)) # number of thermal conductances

        # 动态构建conductance_net
        cond_layers = []
        input_dim = len(input_cols) + self.output_size
        for i, units in enumerate(cond_net_units):
            cond_layers.append(nn.Linear(input_dim, units))
            # 使用当前层的激活函数
            cond_layers.append(get_activation(cond_activations[i]))
            input_dim = units
        cond_layers.append(nn.Linear(input_dim, n_conds))
        cond_layers.append(nn.Sigmoid())
        self.conductance_net = nn.Sequential(*cond_layers)

        # 动态构建ploss_net
        ploss_layers = []
        input_dim = len(input_cols) + self.output_size
        for i, units in enumerate(ploss_net_units):
            ploss_layers.append(nn.Linear(input_dim, units))
            if i < len(ploss_net_units) - 1:
                ploss_layers.append(get_activation(ploss_activations[i]))
            input_dim = units
        ploss_layers.append(nn.Linear(input_dim, self.output_size))
        self.ploss = nn.Sequential(*ploss_layers)

        # 其余初始化代码
        self.adj_mat = np.zeros((n_temps, n_temps), dtype=int)
        triu_idx = np.triu_indices(n_temps, 1)
        adj_idx_arr = np.ones_like(self.adj_mat)
        adj_idx_arr = adj_idx_arr[triu_idx].ravel()
        self.adj_mat[triu_idx] = np.cumsum(adj_idx_arr) - 1
        self.adj_mat += self.adj_mat.T
        self.adj_mat = torch.from_numpy(self.adj_mat[:self.output_size, :]).type(torch.int64)
        self.temp_idcs = [i for i, x in enumerate(input_cols) if x in temperature_cols]
        self.nontemp_idcs = [i for i, x in enumerate(input_cols) if x not in temperature_cols + ["profile_id"]]       

    def forward(self, inp: Tensor, hidden: Tensor) -> Tuple[Tensor, Tensor]:
        prev_out = hidden
        temps = torch.cat([prev_out, inp[:, self.temp_idcs]], dim=1)
        sub_nn_inp = torch.cat([inp, prev_out], dim=1)
        # conducts = torch.abs(self.conductance_net(sub_nn_inp))
        conducts = self.conductance_net(sub_nn_inp)
        # power_loss = torch.abs(self.ploss(sub_nn_inp))
        power_loss = smooth_abs(self.ploss(sub_nn_inp))
        temp_diffs = torch.sum(
            (temps.unsqueeze(1) - prev_out.unsqueeze(-1)) * conducts[:, self.adj_mat],
            dim=-1,
        )
        out = prev_out + self.sample_time * torch.exp(self.caps) * (temp_diffs + power_loss)
        return prev_out, torch.clip(out, -1, 5)  
    
    #%%
# DiffEqLayer定义（支持TorchScript）
class DiffEqLayer(ScriptModule):
    def __init__(self, cell_module):
        super().__init__()
        self.cell = cell_module
        
    @script_method
    def forward(self, input: Tensor, state: Tensor) -> Tuple[Tensor, Tensor]:
        inputs = input.unbind(0)
        outputs = []
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs.append(out)
        return torch.stack(outputs), state
    
#%%
# Optuna目标函数
def objective(trial: optuna.Trial) -> float:
    # 1. 超参数采样
    cond_net_layers = trial.suggest_int('cond_net_layers', 1, 3)
    cond_net_units = [trial.suggest_int(f'cond_net_units_{i}', 2, 30) 
                     for i in range(cond_net_layers)]   
    ploss_net_layers = trial.suggest_int('ploss_net_layers', 1, 3)
    ploss_net_units = [trial.suggest_int(f'ploss_net_units_{i}', 2, 30) 
                      for i in range(ploss_net_layers)]
    cond_activations = [
        trial.suggest_categorical(f'cond_activation_{i}', 
        ["Sigmoid", "tanh", "linear", "ReLU", "biased Elu", "sinus"])
        for i in range(cond_net_layers)
    ]
    # 为ploss_net每层单独采样激活函数
    ploss_activations = [
        trial.suggest_categorical(f'ploss_activation_{i}', 
        ["Sigmoid", "tanh", "linear", "ReLU", "biased Elu", "sinus"])
        for i in range(ploss_net_layers)
    ]    

    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'NAdam'])
    tbptt_size = trial.suggest_int('tbptt_size', 50, 512)

    # 构建模型（传递激活函数列表）
    tnn_cell = TNNCell(
        cond_net_units, cond_activations,  # 传递列表
        ploss_net_units, ploss_activations  # 传递列表
    ).to(device)

    model = DiffEqLayer(tnn_cell).to(device)
    
    # 3. 选择优化器
    if optimizer_name == 'Adam':
        opt = optim.Adam(model.parameters(), lr=lr)
    else:
        opt = optim.NAdam(model.parameters(), lr=lr)

    # 4. 训练与评估（使用训练集本身）
    n_epochs = 100
    loss_func = nn.MSELoss(reduction="none")
    best_train_loss = float('inf')

    for epoch in range(n_epochs):
        hidden = train_tensor[0, :, -len(target_cols):]
        n_batches = int(np.ceil(train_tensor.shape[0] / tbptt_size))
        epoch_loss = 0.0
        
        model.train()
        for i in range(n_batches):
            opt.zero_grad()
            
            output, hidden = model(
                train_tensor[i*tbptt_size : (i+1)*tbptt_size, :, :len(input_cols)],
                hidden.detach().requires_grad_(True)
            )
            
            # 计算训练损失
            loss = loss_func(
                output,
                train_tensor[i*tbptt_size : (i+1)*tbptt_size, :, -len(target_cols):]
            )
            
            # 加权损失计算
            loss = (loss * train_sample_weights[i*tbptt_size:(i+1)*tbptt_size, :, None]).sum() 
            loss /= train_sample_weights[i*tbptt_size:(i+1)*tbptt_size, :].sum() + 1e-8  # 防止除零
            
            # 添加梯度检查（提前拦截问题）[9](@ref)
            if not loss.requires_grad:
                print(f"梯度断裂! Trial: {trial.number}, Batch: {i}")
                raise optuna.TrialPruned()
            # 反向传播
            try:
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
            except RuntimeError as e:
                if "does not have a grad_fn" in str(e):
                    raise optuna.TrialPruned()

        # 计算平均训练损失
        avg_epoch_loss = epoch_loss / n_batches
        
        # 报告训练损失给Optuna
        trial.report(avg_epoch_loss, epoch)
        
        # 剪枝逻辑（基于训练损失）
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        # 记录最佳训练损失
        if avg_epoch_loss < best_train_loss:
            best_train_loss = avg_epoch_loss
    
    return best_train_loss

#%%
# 配置共享存储（SQLite或MySQL）
storage_name = "sqlite:///optuna.db"
study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(),
    storage=storage_name,
    study_name="tnn_optimization",
    load_if_exists=True
)

with parallel_backend('multiprocessing', n_jobs=8):  # Overrides `prefer="threads"` to use multi-processing.
    study.optimize(objective, n_trials=100, show_progress_bar=True)

# # 并行执行（n_jobs为并行进程数）
# with parallel_backend("threading", n_jobs=1):  # 或使用多进程"multiprocessing"
#     study.optimize(objective, n_trials=1000, show_progress_bar=True)

# from multiprocessing import Pool
# from multiprocessing.pool import Pool


# def run_optimization(_):
#     study = optuna.create_study(
#         direction="minimize",
#         sampler=optuna.samplers.TPESampler(),
#         storage=storage_name,
#         study_name="tnn_optimization",
#         load_if_exists=True
#     )

#     study.optimize(objective, n_trials=3)
# if __name__ == "__main__":
#     with Pool(processes=8) as pool:
#         pool.map(run_optimization, range(300))