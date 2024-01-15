# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.env import VecEnv


class LSTM(torch.nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = torch.nn.LSTM(
            input_size=60,
            hidden_size=64,
            num_layers=3,
            batch_first=True
        )
        self.out = torch.nn.Linear(64, 12)

        for name, param in self.lstm.named_parameters():
            torch.nn.init.uniform(param, -10, 10)

    def forward(self, x, h0, c0):
        # shape
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, output_size)
        r_out, (hn, cn) = self.lstm(x, (h0, c0))
        outs = []
        for time_step in range(r_out.size(1)):
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), (hn, cn)


class OnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        self.N_mid = -1024

        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs

        actor_critic_class = eval(self.cfg["policy_class_name"])  # ActorCritic
        actor_critic: ActorCritic = actor_critic_class(self.env.num_obs,
                                                       num_critic_obs,
                                                       self.env.num_actions + 10,
                                                       **self.policy_cfg).to(self.device)
        actor_critic_bro: ActorCritic = actor_critic_class(self.env.num_obs,
                                                           num_critic_obs,
                                                           self.env.num_actions,
                                                           **self.policy_cfg).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"])  # PPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.alg_bro: PPO = alg_class(actor_critic_bro, device=self.device, **self.alg_cfg)

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs + self.N_mid, self.num_steps_per_env, [self.env.num_obs],
                              [self.env.num_privileged_obs], [self.env.num_actions + 10])
        self.alg_bro.init_storage(-self.N_mid, self.num_steps_per_env, [self.env.num_obs],
                                  [self.env.num_privileged_obs], [self.env.num_actions])
        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()

        self.idm_net = LSTM().cuda()
        self.h_state = torch.zeros([3, -self.N_mid, 64], dtype=torch.float32, device=self.device)
        self.c_state = torch.zeros([3, -self.N_mid, 64], dtype=torch.float32, device=self.device)
        self.h_state_eval = torch.zeros([3, self.env.num_envs + self.N_mid, 64], dtype=torch.float32,
                                        device=self.device)
        self.c_state_eval = torch.zeros([3, self.env.num_envs + self.N_mid, 64], dtype=torch.float32,
                                        device=self.device)
        self.optimizer = torch.optim.Adam(self.idm_net.parameters(), lr=0.002)
        self.pred_loss = torch.nn.MSELoss().cuda()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)
        self.alg_bro.actor_critic.train()  # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            obs_buf_idm_array = torch.zeros([-self.N_mid, self.num_steps_per_env, 60], dtype=torch.float32,
                                            device=self.device)
            env_actions_array = torch.zeros([-self.N_mid, self.num_steps_per_env, 12], dtype=torch.float32,
                                            device=self.device)
            with (torch.inference_mode()):
                for i in range(self.num_steps_per_env):
                    obs_next = self.alg.act(obs[:self.N_mid, :], critic_obs[:self.N_mid, :])
                    actions_bro = self.alg_bro.act(obs[self.N_mid:, :], critic_obs[self.N_mid:, :])

                    input = torch.zeros_like(obs_next).to(self.device)
                    input[:, :18] = obs_next[:, :18]
                    input[:, 18:] = (obs_next[:, 18:] > 0.0)
                    idm_obs_buf = torch.zeros([self.env.num_envs + self.N_mid, 1, 60], dtype=torch.float32,
                                              device=self.device)
                    idm_obs_buf[:, 0, :] = torch.cat((obs[:self.N_mid, 9:13],
                                                      obs[:self.N_mid, 16:28],
                                                      obs[:self.N_mid, :3],
                                                      obs[:self.N_mid, 3:6],
                                                      obs[:self.N_mid, 28:40], obs[:self.N_mid, -4:],
                                                      input),
                                                     dim=-1)

                    actions_pred, (self.h_state_eval, self.c_state_eval) = self.idm_net(idm_obs_buf,
                                                                                        self.h_state_eval,
                                                                                        self.c_state_eval)

                    actions = torch.cat((actions_pred[:, 0, :], actions_bro), dim=0)
                    obs, privileged_obs, rewards, dones, infos, obs_buf_idm, env_actions = self.env.step(actions)

                    obs_buf_idm_array[:, i, :] = obs_buf_idm[self.N_mid:, 0, :]
                    env_actions_array[:, i, :] = env_actions[self.N_mid:, :]

                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(
                        self.device), dones.to(self.device)

                    info1 = None
                    info2 = None
                    if 'time_outs' in infos:
                        info1 = {'time_outs': infos['time_outs'][:self.N_mid]}
                        info2 = {'time_outs': infos['time_outs'][self.N_mid:]}

                    self.alg.process_env_step(rewards[:self.N_mid], dones[:self.N_mid], info1)
                    self.alg_bro.process_env_step(rewards[self.N_mid:], dones[self.N_mid:], info2)

                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs[:self.N_mid, :])
                self.alg_bro.compute_returns(critic_obs[self.N_mid:, :])

            for k in range(30):
                actions_pred, (self.h_state, self.c_state) = self.idm_net(obs_buf_idm_array, self.h_state, self.c_state)
                self.h_state = torch.autograd.Variable(self.h_state.data).cuda()
                self.c_state = torch.autograd.Variable(self.c_state.data).cuda()
                loss = self.pred_loss(10.0 * actions_pred, 10.0 * env_actions_array)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if k == 29:
                    print(f'idm loss: {loss}')

            mean_value_loss, mean_surrogate_loss = self.alg.update()
            mean_value_loss_bro, mean_surrogate_loss_bro = self.alg_bro.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))

                model_scripted = torch.jit.script(self.idm_net)  # Export to TorchScript
                model_scripted.save(
                    os.path.join(self.log_dir, 'idm_{}.pt'.format(it)))  # Save

            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

        model_scripted = torch.jit.script(self.idm_net)  # Export to TorchScript
        model_scripted.save(os.path.join(self.log_dir, 'idm_{}.pt'.format(self.current_learning_iteration)))  # Save

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
        }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
