import pandas as pd
import random
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import optuna
import json
from torch import Tensor
from tqdm import tqdm
from typing import List, Tuple
from torch.jit import ScriptModule, script_method
from pathlib import Path
from torch.optim.lr_scheduler import LambdaLR
from joblib import parallel_backend
#%%

configs = json.load(open('configs.json', 'r'))
device = configs['device']

#%%
p = Path(r'data')
train_tensor = torch.load(p / 'train_tensor.pt').to(device)
train_sample_weights = torch.load(p / 'train_sample_weights.pt').to(device)
test_tensor = torch.load(p / 'test_tensor.pt').to(device)
test_sample_weights = torch.load(p / 'test_sample_weights.pt').to(device)
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

def get_optimizer(optimizer_name: str, parameters, lr: float) -> optim.Optimizer:
    optimizer_dict = {
        "Adam": optim.Adam,
        "NAdam": optim.NAdam
    }
    return optimizer_dict[optimizer_name](parameters, lr=lr)

def get_loss_fn(loss_name: str) -> nn.Module:
    loss_dict = {
        "MSE": nn.MSELoss(),
    }
    return loss_dict[loss_name]

#%%
# TNNCell定义（支持动态结构和激活函数）
class TNNCell(nn.Module):
    def __init__(
                self, 
                cond_net_units, 
                cond_activations, 
                ploss_net_units, 
                ploss_activations, 
                sample_time, 
                input_cols, 
                target_cols, 
                temperature_cols):
        
        super().__init__()
        self.sample_time = sample_time
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

    def forward(self, inp: Tensor, hidden: Tensor) -> Tuple[Tensor, Tensor]:
        prev_out = hidden
        temps = torch.cat([prev_out, inp[:, self.temp_idcs]], dim=1)
        sub_nn_inp = torch.cat([inp, prev_out], dim=1)
        conducts = self.conductance_net(sub_nn_inp)
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
def train_nn(**kwargs):
    model = kwargs.pop("model", None)
    cond_net_layers = kwargs.pop("cond_net_layers", 3)
    cond_net_units = kwargs.pop("cond_net_units", [20] * cond_net_layers)
    ploss_net_layers = kwargs.pop("ploss_net_layers", 2)
    ploss_net_units = kwargs.pop("ploss_net_units", [20] * ploss_net_layers)
    cond_activations = kwargs.pop("cond_activations", ["Sigmoid"] * cond_net_layers)
    ploss_activations = kwargs.pop("ploss_activations", ["tanh"] * ploss_net_layers)
    sample_time = kwargs.pop("sample_time", configs['sample_time'])
    input_cols = kwargs.pop("input_cols", configs['input_cols'])
    target_cols = kwargs.pop("target_cols", configs['target_cols'])
    temperature_cols = kwargs.pop("temperature_cols", configs['temperature_cols'])
    lr = kwargs.pop("lr", 1e-3)
    lr_factor = kwargs.pop("lr_factor", 1e-2)
    optimizer_name = kwargs.pop("optimizer", "Adam")
    loss_name = kwargs.pop("loss_name", "MSE")
    tbptt_size = kwargs.pop("tbptt_size", 512)
    n_epochs = kwargs.pop("n_epochs")
    trial = kwargs.pop("trial", None)
    device = kwargs.pop("device", "cpu")

    if not model:
        # 构建模型（传递激活函数列表）
        tnn_cell = TNNCell(
            cond_net_units, cond_activations,  # 传递列表
            ploss_net_units, ploss_activations,  # 传递列表
            sample_time, input_cols, target_cols, temperature_cols
        ).to(device)
        model = DiffEqLayer(tnn_cell).to(device)

    # 3. 选择优化器
    opt = get_optimizer(optimizer_name, model.parameters(), lr)

    loss_func = get_loss_fn(loss_name)
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
            loss = (loss * train_sample_weights[i*tbptt_size : (i+1)*tbptt_size, :, None])
            loss /= train_sample_weights[i*tbptt_size : (i+1)*tbptt_size, :].sum()
            loss = loss.sum().mean()  # 先求和再平均
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
        lr = avg_epoch_loss * lr_factor  # 简单的线性调整学习率
        for param_group in opt.param_groups:
            param_group['lr'] = lr
        
        if not trial:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_epoch_loss:.6f}, lr: {lr:.6f}")
        # 报告训练损失给Optuna
        if trial is not None:
            trial.report(avg_epoch_loss, epoch)
        
        # 剪枝逻辑（基于训练损失）
        if trial is not None and trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        # 记录最佳训练损失
        if avg_epoch_loss < best_train_loss:
            best_train_loss = avg_epoch_loss   
        if not trial and (epoch + 1) % 50 == 0:
            
            # model saving and loading
            mdl_path = Path('models')
            mdl_path.mkdir(exist_ok=True, parents=False)
            mdl_file_path = mdl_path / 'tnn_model.pt'
            model.save(mdl_file_path)  # save
            print(f"Model saved at epoch {epoch+1} with training loss {avg_epoch_loss:.6f}")
    if trial is not None:
        return best_train_loss      
    else:
        return model
    
#%%

def objective(trial: optuna.Trial) -> float:
    # 1. 超参数采样
    cond_net_layers = trial.suggest_int('cond_net_layers', *configs['cond_net_layer'])
    cond_net_units = [trial.suggest_int(f'cond_net_units_{i}', *configs['cond_net_units']) 
                     for i in range(cond_net_layers)]   
    ploss_net_layers = trial.suggest_int('ploss_net_layers', *configs['ploss_net_layers'])
    ploss_net_units = [trial.suggest_int(f'ploss_net_units_{i}', *configs['ploss_net_units']) 
                      for i in range(ploss_net_layers)]
    cond_activations = [
        trial.suggest_categorical(f'cond_activation_{i}', 
        configs['cond_activations'])
        for i in range(cond_net_layers)
    ]
    # 为ploss_net每层单独采样激活函数
    ploss_activations = [
        trial.suggest_categorical(f'ploss_activation_{i}', 
        configs['ploss_activations'])
        for i in range(ploss_net_layers)
    ]    

    lr = 1e-3  # 固定初始学习率，训练过程中动态调整
    lr_factor = trial.suggest_float('lr_factor', *configs['lr_factor'])
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'NAdam'])
    loss_name = trial.suggest_categorical('loss_name', configs['loss_name'])
    tbptt_size = trial.suggest_int('tbptt_size', *configs["tbptt_size"])    
    n_epochs = configs['n_epochs']  # 固定训练轮数
    # 2. 模型训练
    best_train_loss = train_nn(
        cond_net_layers=cond_net_layers,
        cond_net_units=cond_net_units,
        ploss_net_layers=ploss_net_layers,
        ploss_net_units=ploss_net_units,
        cond_activations=cond_activations,
        ploss_activations=ploss_activations,
        lr=lr,
        lr_factor=lr_factor,
        optimizer=optimizer_name,
        loss_name=loss_name,
        tbptt_size=tbptt_size,
        n_epochs=n_epochs,
        trial=trial,
        device=device
    )
    return best_train_loss

if __name__ == "__main__":

    if configs['run_mode'] == 'optimize':
        # 配置共享存储（SQLite或MySQL）
        storage_name = "sqlite:///optuna.db"
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(),
            storage=storage_name,
            study_name="tnn_optimization",
            load_if_exists=True
        )
        print(f"Starting optimization with {configs['n_epochs']} epochs, {configs['n_jobs']} jobs and {configs['n_trials']} trials...")
        with parallel_backend('multiprocessing', n_jobs=configs['n_jobs']):  # Overrides `prefer="threads"` to use multi-processing.
            study.optimize(objective, n_trials=configs['n_trials'], show_progress_bar=True)
    else:
        if Path('models/tnn_model.pt').exists():
            model = torch.jit.load('models/tnn_model.pt')  # load
        else: 
            model = None
        model = train_nn(model=model, n_epochs=configs['n_epochs'], lr=configs['lr'], device=device)
