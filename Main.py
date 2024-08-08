import numpy as np
import torch
#import os


from Environment import Env
#import copy
import json
#import time
#import datetime
#import random
import matplotlib.pyplot as plt
import argparse
#import scipy.special
#import math
#from TD3 import Agent_TD3
#from utils_TD3 import ReplayBuffer
from A3C_core import SharedAdam, ActorCritic

np.random.seed(8888)

parser = argparse.ArgumentParser()
parser.add_argument("--method", default="DDPG_MC")  # Policy name
# parser.add_argument("--env_name", default="LunarLanderContinuous-v2")  # OpenAI gym environment name
parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
parser.add_argument("--start_timesteps", default=1e4, type=int)  # How many time steps purely random policy is run for
parser.add_argument("--eval_freq", default=1e3, type=float)  # How often (time steps) we evaluate
parser.add_argument("--max_timesteps", default=1e6, type=float)  # Max time steps to run environment for
parser.add_argument("--save_models", action="store_true")  # Whether or not models are saved
parser.add_argument("--expl_noise", default=0.0, type=float)  # Std of Gaussian exploration noise
parser.add_argument("--batch_size", default=64, type=int)  # Batch size for both actor and critic
parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
parser.add_argument('--actor_lr', type=float, default=1e-3)  # learning rate
parser.add_argument('--critic_lr', type=float, default=1e-3)  # learning rate
parser.add_argument('--aux_lr', type=float, default=1e-3)  # learning rate
parser.add_argument('--weight_decay', type=float, default=1e-4)  # weight decay
parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noiseparser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
parser.add_argument("--gpu_id", default=0, type=int)  # The id of GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parser.add_argument('-c1', '--config_path1', type=str, help='path to existing scenarios file')
parser.add_argument('-c2', '--config_path2', type=str, help='path to existing options file')
args = parser.parse_args()

#load data
with open("config1.json") as f:
    data_net = json.load(f)

with open("config2.json") as g:
    data_env = json.load(g)

class Data_Class_env:
    def __init__(self, dic:dict[str, float]):
        for key, value in dic.items():
            setattr(self, key, value)

data_env = Data_Class_env(data_env)

class Data_Class_net:
    def __init__(self, dic:dict[str, float]):
        for key, value in dic.items():
            setattr(self, key, value)

data_net = Data_Class_net(data_net)


action_size = 2 * (data_env.M) * (data_env.M + data_env.K) + data_env.N
state_size = [2 * ((data_env.M * data_env.K) + (data_env.N * data_env.K) + (data_env.M * data_env.T) + (data_env.N * data_env.T) + (data_env.N * data_env.M)) + 3]

max_action = 1.0
#replay_buffer = Utils.ReplayBuffer()
training_steps=30
maximum_steps=125

lr_2 = 0.001
noise_clip_2 = 0.5
policy_delay_2 = 2
policy_noise_2 = 0.2 # target policy smoothing noise
batch_size_2 = 64
gamma_2 = 0.99
polyak_2 = 0.995
#filename_2 = "IRS{}".format(n_reflector)
directory_2 = "./SaveWeights"

class main():
    def __init__(self, T):
        self.T = T
        self.EE = 0
        self.snr = 0
        self.R = np.zeros(2)
        self.w = 0
        self.EE_mem = []
        self.snr_mem = []
        self.R_mem = []
        self.reward_mem = []
        self.reward_mem_step = []
        self.t = 0
        self.w_mem = []
        self.env = Env(data_net, data_env, *state_size, action_size)
        #self.agent = DDPG.Agent(alpha=0.0001,beta=0.001,input_dims=state_size,tau=0.005,n_actions=action_size,gamma=0.99,max_size=1000000,C_fc1_dims=1024,C_fc2_dims=512,C_fc3_dims=256,A_fc1_dims=1024,A_fc2_dims=512,batch_size=64,n_agents=1)
        #self.TD3 = Agent_TD3(lr_2, state_size, action_size, max_action=1)
        self.global_actor_critic = ActorCritic(state_size, action_size)
        self.n_worker = 3
        self.workers = []
        for i in range(self.n_worker):
            local_actor_critic = ActorCritic(state_size, action_size, gamma_2)
            self.workers.append(local_actor_critic)
        self.global_actor_critic.share_memory()
        self.optimizer = SharedAdam(self.global_actor_critic.parameters(), lr=lr_2, betas=(0.92, 0.999))
        # self.replay_buffer = ReplayBuffer()
        self.var_noise = 1
        #self.decay_noise = 0.99993
        #self.decay_noise = 0.99933
        self.decay_noise = 0
        #self.decay_noise = 0.99833


    def _(self):
        for e in range(self.T):
            sum_dc = 0
            sum_rc = 0
            sum_dr = 0
            sum_rr = 0
            sum_G = 0
            reward = 0
            reward_t = 0
            EE_t = 0
            R_t = 0
            snr_t = 0
            nstep = 0
            w_t = 0
            w_t_list = []
            snr_t_list = []
            check_snr_t = 0
            check_w_t = 0
            check_w_t_2 = 0
            check_snr_t_10 = 0
            flag = False
            snr_max = 1
            state = self.env.reset()
            sum_dc, h_dc = self.env.gen_h_dc(sum_dc)
            sum_rc, h_rc = self.env.gen_h_rc(sum_rc)
            sum_dr, h_dr = self.env.gen_h_dr(sum_dr)
            sum_rr, h_rr = self.env.gen_h_rr(sum_rr)
            sum_G, G = self.env.gen_G(sum_G)
            state = self.env.state_cal(self.R, self.snr, h_dc, h_rc, h_dr, h_rr, G, snr_max,
                                       np.zeros((action_size, 1)))
            state = abs(state)
            for i in range(self.n_worker):
                self.workers[i].clear_memory()
                while not flag and nstep <= maximum_steps:
                    state = np.reshape(state, (1, *state_size))
                    sum_dc, h_dc = self.env.gen_h_dc(sum_dc)
                    sum_rc, h_rc = self.env.gen_h_rc(sum_rc)
                    sum_dr, h_dr = self.env.gen_h_dr(sum_dr)
                    sum_rr, h_rr = self.env.gen_h_rr(sum_rr)
                    sum_G, G = self.env.gen_G(sum_G)

                    self.var_noise = self.var_noise * self.decay_noise
                    # action = self.agent.choose_action(state)
                    #action = self.TD3.select_action(state)
                    action = self.workers[i].choose_action(state)
                    action = action + (np.random.randn(action_size) * self.var_noise)
                    ahmad = np.min(action)
                    action = action - np.min(action)
                    action = action / np.max(action)
                    w, phi = self.env.action_cal(action)
                    Theta = self.env.gen_Theta(phi)

                    t = nstep % data_env.T
                    self.snr = self.env.cal_snr_r(Theta, w, G, h_dr, h_rr, t)
                    snr_t_list.append(self.snr)
                    snr_max = max(snr_t_list)

                    new_state, reward_t, done, check_snr, check_w, com_w_step, self.R, self.EE = self.env.step(w, phi,
                                                                                                               h_dc,
                                                                                                               h_rc,
                                                                                                               h_dr,
                                                                                                               h_rr, G,
                                                                                                               self.snr,
                                                                                                               snr_max,
                                                                                                               action)

                    EE_t += self.EE
                    R_t += np.sum(self.R) / 2
                    snr_t += self.snr
                    w_t += com_w_step
                    w_t_list.append(com_w_step)

                    if check_snr == 1:
                        check_snr_t += 1
                    if check_w == 1:
                        check_w_t += 1
                    if check_w == 2:
                        check_w_t_2 += 1
                    if check_snr == 10:
                        check_snr_t_10 += 1

                    reward += reward_t
                    self.reward_mem_step.append(reward_t)
                    new_state = abs(new_state)
                    # state_1 = np.reshape(state, (state_size))
                    new_state = np.reshape(new_state, (1, *state_size))
                    # new_state_1 = np.reshape(new_state, (state_size))

                    # self.agent.remember(state, action, reward_t, new_state, done)
                    self.workers[i].remember(state, action, reward_t)
                    if nstep == (maximum_steps - 1):
                        loss = self.workers[i].calc_loss(done = True)
                        self.optimizer.zero_grad()
                        loss.backward()
                        for local_param, global_param in zip(self.workers[i].parameters(),
                                                             self.global_actor_critic.parameters()):
                            global_param._grad = local_param.grad
                        self.optimizer.step()
                        self.workers[i].load_state_dict(self.global_actor_critic.state_dict())

                    #self.replay_buffer.add((state_1, action, reward_t, new_state_1, done))
                    state = new_state.copy()
                    td3_update_or_not = nstep % 5
                    # if td3_update_or_not == 0:
                    #     self.TD3.update(self.replay_buffer, maximum_steps, batch_size_2, gamma_2, polyak_2,
                    #                     policy_noise_2,
                    #                     noise_clip_2, policy_delay_2)
                    # self.agent.learn()
                    # print('step:', self.t, 'reward', self.reward_mem_step[self.t])
                    self.t += 1
                    nstep += 1



            self.reward_mem.append(reward/maximum_steps)
            self.EE_mem.append(EE_t/maximum_steps)
            self.R_mem.append(R_t/maximum_steps)
            self.snr_mem.append(snr_t/maximum_steps)
            self.w_mem.append(w_t/maximum_steps)
            print('episode:', e, 'reward', self.reward_mem[e])
            #print check  state reward not binary   randomness
            print('number of times that w constraint has been satisfied:',check_w_t, 'number of times that snr constraint has been satisfied:', check_snr_t)
            print('number of times that w2 constraint has been satisfied:', check_w_t_2,
                  'number of times that snr10 constraint has been satisfied:', check_snr_t_10)
            print('mean power: ', np.mean(w_t_list))
            print('mean snr: ', np.mean(snr_t_list))
            print('average rate', self.R_mem[e])
            print(sum_G + sum_dc + sum_dr + sum_rc + sum_rr)






        return self.EE_mem, self.R_mem, self.snr_mem, self.reward_mem, self.w_mem, self.reward_mem_step

T = 5000
learning_class = main(T)
EE_mem, R_mem, snr_mem, reward_mem, w_mem, reward_mem_step = learning_class._()

# To smooth the rewards ---------------------    aaa=len(reward_episode)
w_2 = int(T/8)
mean_ep_rewardall= np.zeros(T-w_2)
for i in range(T-w_2) :
    temp_value= np.sum(reward_mem[i:-T + w_2+i])/w_2
    mean_ep_rewardall[i] = temp_value
plt.plot(mean_ep_rewardall)
plt.xlabel('episode')
plt.ylabel('mean reward')
plt.title('mean reward in each episode')
plt.show()

print('EE with 4 RIS elements: ', np.mean(EE_mem[-10:]))
print('mean rate with 4 RIS elements: ', np.mean(R_mem[-10:]))

plt.plot(reward_mem)
plt.xlabel('episode')
plt.ylabel('mean reward')
plt.title('mean reward in each episode')
plt.show()

plt.plot(reward_mem_step)
plt.xlabel('step')
plt.ylabel('reward')
plt.title('reward in each step')
plt.show()


















