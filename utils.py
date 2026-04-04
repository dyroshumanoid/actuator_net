# Modified from:
# https://github.com/Improbable-AI/walk-these-ways/blob/master/scripts/actuator_net/utils.py
# By Gary Lvov
import os
import pickle as pkl
from matplotlib import pyplot as plt
import time
import imageio
import numpy as np
from tqdm import tqdm
from glob import glob
import pickle
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam

from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import ParameterGrid
import datetime
import json

class ActuatorDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['joint_states'])

    def __getitem__(self, idx):
        return {k: v[idx] for k,v in self.data.items()}

class Act(nn.Module):
    def __init__(self, act, slope=0.05):
        super(Act, self).__init__()
        self.act = act
        self.slope = slope
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, input):
        if self.act == "relu":
            return F.relu(input)
        elif self.act == "leaky_relu":
            return F.leaky_relu(input)
        elif self.act == "sp":
            return F.softplus(input, beta=1.)
        elif self.act == "leaky_sp":
            return F.softplus(input, beta=1.) - self.slope * F.relu(-input)
        elif self.act == "elu":
            return F.elu(input, alpha=1.)
        elif self.act == "leaky_elu":
            return F.elu(input, alpha=1.) - self.slope * F.relu(-input)
        elif self.act == "ssp":
            return F.softplus(input, beta=1.) - self.shift
        elif self.act == "leaky_ssp":
            return (
                F.softplus(input, beta=1.) -
                self.slope * F.relu(-input) -
                self.shift
            )
        elif self.act == "tanh":
            return torch.tanh(input)
        elif self.act == "leaky_tanh":
            return torch.tanh(input) + self.slope * input
        elif self.act == "swish":
            return torch.sigmoid(input) * input
        elif self.act == "softsign":
            return F.softsign(input)
        else:
            raise RuntimeError(f"Undefined activation called {self.act}")

def build_mlp(in_dim, units, layers, out_dim, act='relu', layer_norm=False, act_final=False):
    mods = [nn.Linear(in_dim, units), Act(act)]
    for _ in range(layers-1):
        mods += [nn.Linear(units, units), Act(act)]
    mods += [nn.Linear(units, out_dim)]
    if act_final:
        mods += [Act(act)]
    if layer_norm:
        mods += [nn.LayerNorm(out_dim)]
    return nn.Sequential(*mods)

class LSTMModel(nn.Module):

    def __init__(self, input_dim, hidden_size, num_layers, out_dim):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def build_lstm(in_dim, units, layers, out_dim):
    return LSTMModel(in_dim, units, layers, out_dim)

def save_dataloaders(train_loader, test_loader, save_path):
    dataloaders = {'train': train_loader, 'test': test_loader}
    with open(save_path, 'wb') as f:
        pkl.dump(dataloaders, f)

def load_dataloaders(load_path):
    with open(load_path, 'rb') as f:
        dataloaders = pkl.load(f)
    return dataloaders['train'], dataloaders['test']

def train_actuator_network(xs, ys, batch_size, num_samples_in_history, units, layers, lr, epochs, eps, weight_decay,
                           actuator_network_path, dataloader_path, model_type, num_joints=1,
                           pretrained_model_path=None, save_dataloaders_flag=True, return_stats=False,
                           global_step_offset=0, log_dir=None):
    print(xs.shape, ys.shape)
    num_data = xs.shape[0]
    num_train = num_data // 5 * 4
    num_test = num_data - num_train

    dataset = ActuatorDataset({"joint_states": xs, "tau_ests": ys})
    train_set, val_set = random_split(dataset, [num_train, num_test])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    if save_dataloaders_flag:
        save_dataloaders(train_loader, test_loader, dataloader_path)
    
    if pretrained_model_path is not None and os.path.exists(pretrained_model_path):
        model = torch.jit.load(pretrained_model_path)
        print(f"Warm-start from {pretrained_model_path}")
    elif model_type == "mlp":
        model = build_mlp(in_dim=(num_samples_in_history + 1) * 2 * num_joints,
                        units=units, layers=layers, out_dim=num_joints, act='softsign')
    elif model_type == "lstm":
        model = build_lstm(1, units=units, layers=layers, out_dim=num_joints)

    opt = Adam(model.parameters(), lr=lr, eps=eps, weight_decay=weight_decay)
    device = 'cuda:0'
    model = model.to(device)

    if log_dir is None:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%I-%M%p")
        log_dir = f'./logs/bs{batch_size}_u{units}_l{layers}_lr{lr}_eps{eps}_wd{weight_decay}_ns{num_samples_in_history}_{current_time}'
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # Hyperparameters dict for TensorBoard
    hparams = {
        'lr': lr,
        'batch_size': batch_size,
        'units': units,
        'layers': layers,
        'eps': eps,
        'weight_decay': weight_decay,
        'num_samples_in_history': num_samples_in_history
    }

    # Empty dict for any metrics you might want to track along with hyperparameters
    metrics = {
        'test_loss': 0,
        'mae': 0
    }

    # Start with logging the hyperparameters and initial metrics
    writer.add_hparams(hparams, metrics)

    for epoch in range(epochs):
        epoch_loss = 0
        ct = 0
        for batch in train_loader:
            data = batch['joint_states'].to(device)
            if model_type == 'lstm':
                data = data.view(data.size(0), data.size(1), 1) 
            y_pred = model(data)
            opt.zero_grad()
            y_label = batch['tau_ests'].to(device)
            loss = ((y_pred - y_label) ** 2).mean()
            loss.backward()
            opt.step()
            epoch_loss += loss.detach().cpu().numpy()
            ct += 1
        epoch_loss /= ct

        test_loss = 0
        mae = 0
        ct = 0
        with torch.no_grad():
            for batch in test_loader:
                data = batch['joint_states'].to(device)
                if model_type == 'lstm':
                    data = data.view(data.size(0), data.size(1), 1) 
                y_pred = model(data)
                y_label = batch['tau_ests'].to(device)
                tau_est_loss = ((y_pred - y_label) ** 2).mean()
                loss = tau_est_loss
                test_mae = (y_pred - y_label).abs().mean()

                test_loss += loss
                mae += test_mae
                ct += 1
            test_loss /= ct
            mae /= ct

        # Log losses and MAE for each epoch within the same hparam context
        global_step = global_step_offset + epoch
        writer.add_scalar('Loss/train', epoch_loss, global_step)
        writer.add_scalar('Loss/test', test_loss, global_step)
        writer.add_scalar('MAE/test', mae, global_step)

        print(f'epoch: {epoch} | loss: {epoch_loss:.4f} | test loss: {test_loss:.4f} | mae: {mae:.4f}')

        model_scripted = torch.jit.script(model)  # Export to TorchScript
        model_scripted.save(actuator_network_path)  # Save
        
    metrics['test_loss'] = test_loss
    metrics['mae'] = mae
    writer.add_hparams(hparams, metrics)
    writer.close()
    
    if return_stats:
        return model, test_loss.to('cpu').numpy(), mae.to('cpu').numpy()
    else:
        return model

# 12 individual actuator groups (one network per joint, no coupling).
# Indices refer to joint_position_errors/velocities/tau_ests columns (0-based, after platform is stripped).
JOINT_GROUPS = [
    ([0],  "left_hip_roll"),
    ([1],  "left_hip_pitch"),
    ([2],  "left_hip_yaw"),
    ([3],  "left_knee_pitch"),
    ([4],  "left_ankle_pitch"),
    ([5],  "left_ankle_roll"),
    ([6],  "right_hip_roll"),
    ([7],  "right_hip_pitch"),
    ([8],  "right_hip_yaw"),
    ([9],  "right_knee_pitch"),
    ([10], "right_ankle_pitch"),
    ([11], "right_ankle_roll"),
]

EVAL_PKL_NAME = "data_period2.0_radius_0.03.pkl"

def load_experiments(exp_dir,
                    torque_scaling=.01,
                    exclude=None):
    datas = []

    experiments = glob(f"{exp_dir}/*.pkl")
    if exclude:
        experiments = [e for e in experiments if os.path.basename(e) not in exclude]
    for experiment in experiments:
        with open(experiment, 'rb') as f:
            data = pickle.load(f)
            # Maybe good to delineate trials somehow like this: (or better to just run all trials at once)
            # data.extend({
            # "joint_names": ["platform", "pitch", "roll"],
            # "joint_positions": [np.nan, np.nan, np.nan],
            # "joint_velocities": [np.nan, np.nan, np.nan],
            # "joint_efforts": [np.nan, np.nan, np.nan],
            # "time_sec": np.nan,
            # "time_nsec": np.nan,
            # "joint_position_command": [np.nan, np.nan, np.nan],
            #  })
            datas.extend(data) # shamelessly combine trials (messes with transitions
                               # between experiments, but messed up transitions have far less samples than other transitions
                               # so it should be fine. )

    # TODO: for now, we ignore the platform joint, needs to be added back eventually
    num_actuators = len(datas[0]["joint_positions"]) - 1

    tau_ests = np.zeros((len(datas), num_actuators))
    joint_positions = np.zeros((len(datas), num_actuators))
    joint_position_targets = np.zeros((len(datas), num_actuators))
    joint_velocities = np.zeros((len(datas), num_actuators))

    for i in range(len(datas)):
        # TODO: For now, we ignore platform joint. Needs to be added back
        tau_ests[i, :] = np.array(datas[i]["joint_efforts"][1:]) * torque_scaling
        joint_positions[i, :] = datas[i]["joint_positions"][1:]
        joint_position_targets[i, :] = datas[i]["joint_position_command"][1:]
        joint_velocities[i, :] = datas[i]["joint_velocities"][1:]

    joint_position_errors = joint_position_targets - joint_positions
    joint_velocities = joint_velocities

    joint_position_errors = torch.tensor(joint_position_errors, dtype=torch.float)
    joint_velocities = torch.tensor(joint_velocities, dtype=torch.float)
    tau_ests = torch.tensor(tau_ests, dtype=torch.float)

    return joint_position_errors, joint_velocities, tau_ests, num_actuators

def load_single_experiment(pkl_path, torque_scaling=0.01):
    """Load a single pkl file and return (joint_position_errors, joint_velocities, tau_ests)."""
    with open(pkl_path, 'rb') as f:
        datas = pickle.load(f)
    num_actuators = len(datas[0]["joint_positions"]) - 1
    tau_ests        = np.zeros((len(datas), num_actuators))
    joint_positions = np.zeros((len(datas), num_actuators))
    joint_targets   = np.zeros((len(datas), num_actuators))
    joint_velocities= np.zeros((len(datas), num_actuators))
    for i in range(len(datas)):
        tau_ests[i, :]         = np.array(datas[i]["joint_efforts"][1:]) * torque_scaling
        joint_positions[i, :]  = datas[i]["joint_positions"][1:]
        joint_targets[i, :]    = datas[i]["joint_position_command"][1:]
        joint_velocities[i, :] = datas[i]["joint_velocities"][1:]
    jpe = torch.tensor(joint_targets - joint_positions, dtype=torch.float)
    jv  = torch.tensor(joint_velocities,                dtype=torch.float)
    te  = torch.tensor(tau_ests,                        dtype=torch.float)
    return jpe, jv, te

def prepare_data_for_model(joint_position_errors, joint_velocities, tau_ests, num_actuators, num_samples_in_history):
    xs = []
    ys = []
    
    num_samples_in_history += 1 # Include current time step
    # Loop over each actuator
    for i in range(num_actuators):
        # Create list to hold time-shifted features for current actuator
        xs_joint = []

        # Append all position errors first [e(t), e(t-1), ..., e(t-N)], then all velocities [v(t), v(t-1), ..., v(t-N)]
        for t in range(num_samples_in_history-1, -1, -1):
            xs_joint.append(joint_position_errors[t:-(num_samples_in_history-t) if num_samples_in_history-t != 0 else None, i:i+1])
        for t in range(num_samples_in_history-1, -1, -1):
            xs_joint.append(joint_velocities[t:-(num_samples_in_history-t) if num_samples_in_history-t != 0 else None, i:i+1])

        # Concatenate all features horizontally (new feature columns)
        xs_joint = torch.cat(xs_joint, dim=1)
        xs.append(xs_joint)

        # The corresponding target (tau_ests) should be aligned with the last feature time step
        tau_ests_joint = tau_ests[num_samples_in_history-1:, i:i+1]
        ys.append(tau_ests_joint)

    # Concatenate all data vertically (stacking different actuators)
    xs = torch.cat(xs, dim=0)
    ys = torch.cat(ys, dim=0)

    return xs[::num_samples_in_history + 1], ys[::num_samples_in_history + 1]

# Data is recorded at 0.001 s; history samples are spaced 0.01 s apart → stride of 10 steps.
HISTORY_STRIDE = 10

def prepare_data_for_joint_group(joint_position_errors, joint_velocities, tau_ests, joint_indices, num_samples_in_history, history_stride=HISTORY_STRIDE):
    """Like prepare_data_for_model but for a specific group of joints (single or coupled).

    History samples are spaced `history_stride` data-steps apart (default 10 × 0.001 s = 0.01 s).

    For n joints in the group, input features are:
      [e_j0(t)..e_j0(t-N*s), v_j0(t)..v_j0(t-N*s), e_j1(...)..., v_j1(...)...]
    giving in_dim = 2 * n * (num_samples_in_history + 1).
    Output ys shape: (T, n).
    """
    H = num_samples_in_history + 1  # include current timestep
    s = history_stride
    xs_parts = []
    for i in joint_indices:
        for t in range(H - 1, -1, -1):
            # t=H-1 → current (offset 0), t=0 → oldest (offset (H-1)*s steps back)
            start = t * s
            end   = -(H - 1 - t) * s if (H - 1 - t) * s != 0 else None
            xs_parts.append(joint_position_errors[start:end, i:i+1])
        for t in range(H - 1, -1, -1):
            start = t * s
            end   = -(H - 1 - t) * s if (H - 1 - t) * s != 0 else None
            xs_parts.append(joint_velocities[start:end, i:i+1])
    xs = torch.cat(xs_parts, dim=1)
    ys = tau_ests[(H - 1) * s:, joint_indices]
    return xs, ys

def train_actuator_network_and_plot_predictions(experiment_dir, actuator_network_path, dataloader_path, model_type, load_pretrained_model=False):
    hyperparam_sweep = False
    best_params_path = os.path.join(os.path.dirname(actuator_network_path) or ".", "best_params.json")
    all_pkl_files = [f for f in sorted(glob(f"{experiment_dir}/*.pkl"))
                     if os.path.basename(f) != EVAL_PKL_NAME]
    num_samples_in_history = 2

    if load_pretrained_model:
        with open(best_params_path) as f:
            saved_best = json.load(f)
        print(f"Loaded best hyperparams from {best_params_path}")
    else:
        saved_best = {}
        if hyperparam_sweep:
            param_grid = {
                'batch_size': [64],
                'units': [32],
                'layers': [2],
                'lr': [8e-4, 8e-3, 1e-4],
                'eps': [1e-8],
                'weight_decay': [0.0, 1e-8],
                'num_samples_in_history': [2],
                'epochs': [200]
            }

            jpe_all, jv_all, te_all, _ = load_experiments(experiment_dir,
                                                           torque_scaling=.01,
                                                           exclude={EVAL_PKL_NAME})
            for joint_indices, group_name in JOINT_GROUPS:
                num_joints = len(joint_indices)
                print(f"\n{'='*60}\nSweep: {group_name}\n{'='*60}")
                sweep_net_path = actuator_network_path.replace(".pt", f"_{group_name}_sweep_temp.pt")
                sweep_dl_path  = dataloader_path.replace(".dataloader", f"_{group_name}_sweep_temp.dataloader")
                results = []
                for params in tqdm(ParameterGrid(param_grid)):
                    try:
                        print(f"Attempting to train with hyperparameters: {params}")
                        train_xs, train_ys = prepare_data_for_joint_group(
                            jpe_all, jv_all, te_all, joint_indices, params['num_samples_in_history'])
                        (_, test_loss, test_mae) = train_actuator_network(
                            train_xs, train_ys,
                            batch_size=params['batch_size'],
                            num_samples_in_history=params['num_samples_in_history'],
                            units=params['units'], layers=params['layers'],
                            lr=params['lr'], epochs=params['epochs'],
                            eps=params['eps'], weight_decay=params['weight_decay'],
                            actuator_network_path=sweep_net_path,
                            dataloader_path=sweep_dl_path,
                            model_type=model_type, num_joints=num_joints,
                            return_stats=True)
                        results.append((params, (float(test_loss), float(test_mae))))
                    except Exception as e:
                        results.append((params, (float('inf'), float('inf'))))
                        print(f"Failed with hyperparameters: {params}")
                        print(e)
                        continue
                for p in [sweep_net_path, sweep_dl_path]:
                    if os.path.exists(p):
                        os.remove(p)
                np.save(f"results_{group_name}.npy", np.array(results, dtype=object))
                best_params = min(results, key=lambda x: x[1][0])
                print(f"Best params based on test loss: {best_params[0]} with loss: {best_params[1][0]} and MAE: {best_params[1][1]}")
                saved_best[group_name] = best_params[0]

            with open(best_params_path, 'w') as f:
                json.dump(saved_best, f, indent=2)
            print(f"Saved best params → {best_params_path}")
        else:
            fixed_params = dict(batch_size=64, num_samples_in_history=2,
                                units=32, layers=3, lr=8e-4, epochs=200,
                                eps=1e-8, weight_decay=0.0)
            for _, group_name in JOINT_GROUPS:
                saved_best[group_name] = fixed_params

        # Sequential training: pkl outer loop, group inner loop
        FINETUNE_EPOCH_RATIO = 0.5

        # Clear any leftover checkpoints so first pkl always starts from scratch
        for _, group_name in JOINT_GROUPS:
            group_net_path = actuator_network_path.replace(".pt", f"_{group_name}.pt")
            if os.path.exists(group_net_path):
                os.remove(group_net_path)

        group_epoch_offset = {group_name: 0 for _, group_name in JOINT_GROUPS}

        for pkl_idx, pkl_path in tqdm(enumerate(all_pkl_files), total=len(all_pkl_files), desc="PKL files"):
            print(f"\n{'='*60}\nPKL {pkl_idx+1}/{len(all_pkl_files)}: {os.path.basename(pkl_path)}\n{'='*60}")
            for joint_indices, group_name in tqdm(JOINT_GROUPS, desc="Joint groups", leave=False):
                params = saved_best.get(group_name)
                if params is None:
                    continue

                num_joints       = len(joint_indices)
                group_net_path   = actuator_network_path.replace(".pt", f"_{group_name}.pt")
                group_dl_path    = dataloader_path.replace(".dataloader", f"_{group_name}.dataloader")
                fine_tune_epochs = max(20, int(params['epochs'] * FINETUNE_EPOCH_RATIO))
                group_log_dir    = f'./logs/{group_name}'

                jpe, jv, te = load_single_experiment(pkl_path, torque_scaling=.01)
                train_xs, train_ys = prepare_data_for_joint_group(
                    jpe, jv, te, joint_indices, params['num_samples_in_history'])

                pretrained = group_net_path if os.path.exists(group_net_path) else None
                train_actuator_network(
                    train_xs, train_ys,
                    batch_size=params['batch_size'],
                    num_samples_in_history=params['num_samples_in_history'],
                    units=params['units'], layers=params['layers'],
                    lr=params['lr'], epochs=fine_tune_epochs,
                    eps=params['eps'], weight_decay=params['weight_decay'],
                    actuator_network_path=group_net_path,
                    dataloader_path=group_dl_path,
                    model_type=model_type, num_joints=num_joints,
                    pretrained_model_path=pretrained,
                    global_step_offset=group_epoch_offset[group_name],
                    log_dir=group_log_dir)

                group_epoch_offset[group_name] += fine_tune_epochs

    print("\nAll groups done.")
    return saved_best