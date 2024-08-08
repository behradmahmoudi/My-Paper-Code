import json
import copy
import torch
import numpy as np
import random

import math

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

np.random.seed(8888)

class Data_Class_net:
    def __init__(self, dic:dict[str, float]):
        for key, value in dic.items():
            setattr(self, key, value)

data_net = Data_Class_net(data_net)
#creat the environment
class Env:
    def __init__(self, data_network, data_environment, state_dim, action_dim):
        self.net_data = data_network
        self.env_data = data_environment
        self.e = np.e
        self.pi = np.pi
        self.state_size = state_dim
        self.action_size = action_dim
        self.action_space = []
        for i in range(self.action_size):
            self.action_space.append(random.uniform(0,1))
        self.h_dc = self.gen_h_dc(0)
        self.h_rc = self.gen_h_rc(0)
        self.G = self.gen_G(0)
        self.pos_BS = [0, 0]
        self.pos_RIS = [50, 0]


    def gen_h_dc(self, sum):
        random_dc = np.random.randn(self.env_data.M, self.env_data.K) + 1j * np.random.randn(self.env_data.M, self.env_data.K)
        if np.linalg.norm(random_dc, ord = 'fro') < 1:
            random_dc = 0.01 * (np.ones((self.env_data.M, self.env_data.K)) + 1j * np.ones((self.env_data.M, self.env_data.K)))
            sum += 1
        #random_dc = np.ones([self.env_data.M, self.env_data.K])
        pos_hdc = np.random.uniform(0, 100, 2)
        #dist = np.sqrt(((pos_hdc[0] - 0)**2) + ((pos_hdc[1] - 0)**2) )
        dist = 20
        #path_loss = 10**(-1 * (12.81 + 3.76*(np.log10(dist/1000))))
        path_loss = np.sqrt((dist**(-3.3)) * (10**(-3)))
        h_dc = random_dc * path_loss
        return sum, h_dc
    def gen_h_rc(self, sum):
        random_rc = np.random.randn(self.env_data.N, self.env_data.K) + 1j * np.random.randn(self.env_data.N, self.env_data.K)
        if np.linalg.norm(random_rc, ord = 'fro') < 1:
            random_dc = 0.01 * (np.ones((self.env_data.N, self.env_data.K)) + 1j * np.ones((self.env_data.N, self.env_data.K)))
            sum += 1
        #random_rc = np.ones([self.env_data.N, self.env_data.K])
        pos_hrc = np.random.uniform(0, 100, 2)
        #dist = np.sqrt(((pos_hrc[0] - 50) ** 2) + ((pos_hrc[1] - 0) ** 2))
        dist = 3
        #path_loss = 10**(-1 * (12.81 + 3.76*(np.log10(dist/1000))))
        path_loss = np.sqrt((dist**(-2.3)) * (10**(-3)))
        h_rc = random_rc * path_loss
        return sum, h_rc

    def gen_h_dr(self, sum):
        random_dr = np.random.randn(self.env_data.M, self.env_data.T) + 1j * np.random.randn(self.env_data.M, self.env_data.T)
        if np.linalg.norm(random_dr, ord = 'fro') < 1:
            random_dr = 0.01 * (np.ones((self.env_data.M, self.env_data.T)) + 1j * np.ones((self.env_data.M, self.env_data.T)))
            sum += 1
        #random_dr = np.ones([self.env_data.M, self.env_data.T])
        pos_hdr = np.random.uniform(0, 100, 2)
        #dist = np.sqrt(((pos_hdr[0] - 0) ** 2) + ((pos_hdr[1] - 0) ** 2))
        dist = 20
        #path_loss = 10**(-1 * (12.81 + 3.76*(np.log10(dist/1000))))
        path_loss = np.sqrt((dist**(-2.7)) * (10**(-3)))
        h_dr = random_dr * path_loss
        return sum, h_dr
    def gen_h_rr(self, sum):
        random_rr = np.random.randn(self.env_data.N, self.env_data.T) + 1j * np.random.randn(self.env_data.N, self.env_data.T)
        if np.linalg.norm(random_rr, ord = 'fro') < 1:
            random_rr = 0.01 * (np.ones((self.env_data.N, self.env_data.T)) + 1j * np.ones((self.env_data.N, self.env_data.T)))
            sum += 1
        #random_rr = np.ones([self.env_data.N, self.env_data.T])
        pos_hrr = np.random.uniform(0, 100, 2)
        #dist = np.sqrt(((pos_hrr[0] - 50) ** 2) + ((pos_hrr[1] - 0) ** 2))
        dist = 3
        #path_loss = 10**(-1 * (12.81 + 3.76*(np.log10(dist/1000))))
        path_loss = np.sqrt((dist**(-2.3)) * (10**(-3)))
        h_rr = random_rr * path_loss
        return sum, h_rr
    def gen_G(self, sum):
        random_g = np.random.randn(self.env_data.N, self.env_data.M) + 1j * np.random.randn(self.env_data.N, self.env_data.M)
        if np.linalg.norm(random_g, ord = 'fro') < 1:
            random_g = 0.01 * (np.ones((self.env_data.N, self.env_data.M)) + 1j * np.ones((self.env_data.N, self.env_data.M)))
            sum += 1
        #random_g = np.ones([self.env_data.N, self.env_data.M])
        pos_ris = [50, 0]
        #dist = np.sqrt(((pos_ris[0] - 0) ** 2) + ((pos_ris[1] - 0) ** 2))
        dist = 30
        #path_loss = 10**(-1 * (12.81 + 3.76*(np.log10(dist/1000))))
        path_loss = np.sqrt((dist**(-2.3)) * (10**(-3)))
        G = random_g * path_loss
        return sum, G

    def gen_Theta(self, phi):
        Theta = np.zeros([self.env_data.N, self.env_data.N], dtype=complex)

        for n in range(self.env_data.N):
            phi_n = phi[n]
            Theta[n, n] = self.e ** (1j*phi_n)
        return Theta

    def angle(self, matrix):
        fd, sd = matrix.shape

        angle = np.zeros((fd, sd))

        for i in range(fd):
            for j in range(sd):
                if np.angle(matrix[i, j]) < 0:
                    angle[i, j] = np.angle(matrix[i, j]) + (2 * np.pi)
                else:
                    angle[i, j] = np.angle(matrix[i, j])

        return angle


    def cal_gamma_c(self, Theta, w, G, h_dc, h_rc):
        gamma = np.zeros([self.env_data.K])
        for k in range(self.env_data.K):
            num = 0
            T = np.matmul(Theta, G)  # N*N * N*M = N*M
            #h = np.transpose(h_rc[:, k]) # 1*N
            h = np.reshape(h_rc[:, k], (1, self.env_data.N))
            num += np.matmul(h, T)  # 1*M
            #num += np.transpose(h_dc[:, k])
            num += np.reshape(h_dc[:, k], (1, self.env_data.M))
            num = np.matmul(num, w[:, k])
            num = (np.linalg.norm(num)) ** 2
            den = []
            for kk in range(self.env_data.K + self.env_data.M):
                if kk!=k:
                    den1 = 0
                    T1 = np.matmul(Theta, G)  # N*N * N*M = N*M
                    #h1 = np.transpose(h_rc[:, k])  # 1*N
                    h1 = np.reshape(h_rc[:, k], (1, self.env_data.N))
                    den1 += np.matmul(h1, T1)  # 1*M
                    #den1 += np.transpose(h_dc[:, k])
                    den1 += np.reshape(h_dc[:, k], (1, self.env_data.M))
                    den1 = np.matmul(den1, w[:, kk])
                    den1 = (np.linalg.norm(den1)) ** 2
                    den.append(den1)
                else:
                    pass
            interference = sum(den)
            noise = self.env_data.sigma
            gamma1 = num/(interference + noise)
            gamma[k] = gamma1
            return gamma


    def cal_snr_r(self, Theta, w, G, h_dr, h_rr, t):
        num = 0
        T = np.matmul(Theta, G)  # N*N * N*M = N*M
        h = np.reshape(h_rr[:, t], (1,self.env_data.N)) # 1*N
        num += np.matmul(h, T)  # 1*M
        num += np.reshape(h_dr[:, t], (1,self.env_data.M))
        h_t = np.matmul(num.T, num)
        h_t_h = h_t.conj().T
        w_h = w.conj().T
        mul1 = np.matmul(h_t, w)
        mul2 = np.matmul(w_h, h_t_h)
        mul3 = np.matmul(mul2, mul1)
        trace = np.trace(mul3)
        noise = self.env_data.sigma
        snr = trace/(noise)
        return np.real(snr)

    def invQ(self):
        return 5.1993375821928165

    def cal_R(self, gamma):
        R = np.zeros(self.env_data.K)
        V = np.zeros(self.env_data.K)
        for k in range(self.env_data.K):
            V[k] = (np.log2(self.e) ** 2) * (1-((1+gamma[k])**(-2)))
            R[k] = np.log2(1 + gamma[k]) - (self.invQ() * math.sqrt(V[k]/self.env_data.md))
        return R

    def EE(self, R, w):
        sum_R = np.sum(R)
        E = 0
        for k in range(self.env_data.K):
            E += np.linalg.norm(w[:, k]) **2
        E += self.env_data.ps + self.env_data.N * self.env_data.pd + self.env_data.pc
        EE = sum_R/E
        return EE

    def state_cal(self, R_1, snr, h_dc, h_rc, h_dr, h_rr, G, snr_max, action):
        state = np.zeros(self.state_size, dtype=complex)

        start = 0   #h_dc
        end = (self.env_data.M * self.env_data.K)
        s1 = np.reshape(abs(h_dc), self.env_data.M * self.env_data.K)
        state[start:end] = s1 * (10**(3))

        start = end  # h_dc
        end = end + (self.env_data.M * self.env_data.K)
        s11 = np.reshape(self.angle(h_dc), self.env_data.M * self.env_data.K)
        state[start:end] = s11

        start = end  # h_rc
        end = end + (self.env_data.N * self.env_data.K)
        s2 = np.reshape(abs(h_rc), self.env_data.N * self.env_data.K)
        state[start:end] = s2 * (10**(3))

        start = end  # h_rc
        end = end + (self.env_data.N * self.env_data.K)
        s22 = np.reshape(self.angle(h_rc), self.env_data.N * self.env_data.K)
        state[start:end] = s22



        start = end  # h_dr
        end = end + (self.env_data.M * self.env_data.T)
        s3 = np.reshape(abs(h_dr), self.env_data.M * self.env_data.T)
        state[start:end] = s3 * (10**(3))

        start = end  # h_dr
        end = end + (self.env_data.M * self.env_data.T)
        s33 = np.reshape(self.angle(h_dr), self.env_data.M * self.env_data.T)
        state[start:end] = s33

        start = end  # h_rr
        end = end + (self.env_data.N * self.env_data.T)
        s4 = np.reshape(abs(h_rr), self.env_data.N * self.env_data.T)
        state[start:end] = s4 * (10**(3))

        start = end  # h_rr
        end = end + (self.env_data.N * self.env_data.T)
        s44 = np.reshape(self.angle(h_rr), self.env_data.N * self.env_data.T)
        state[start:end] = s44

        start = end  # G
        end = end + (self.env_data.N * self.env_data.M)
        s5 = np.reshape(abs(G), self.env_data.N * self.env_data.M)
        state[start:end] = s5 * (10**(3))

        start = end  # G
        end = end + (self.env_data.N * self.env_data.M)
        s55 = np.reshape(self.angle(G), self.env_data.N * self.env_data.M)
        state[start:end] = s55

        start = end
        end = end + 1
        state[start:end] = R_1[0]

        start = end
        end = end + 1
        state[start:end] = R_1[1]

        start = end
        end = end + 1
        state[start:end] = snr

        #start = end
        #end = end + self.action_size
        #state[start:end] = action.reshape(self.action_size)
        return state

    def action_cal(self, action):
        action = action.reshape(self.action_size, 1)
        start = 0
        end = self.env_data.M * (self.env_data.M + self.env_data.K)
        w = np.zeros([self.env_data.M, (self.env_data.K + self.env_data.M)], dtype=complex)
        w1 = (action[start: end]) * (np.sqrt(((self.env_data.pmax)/(self.env_data.M + self.env_data.K)))) * 0.8

        start = end
        end = end + self.env_data.M * (self.env_data.M + self.env_data.K)
        w2 = (action[start: end])

        w1 = np.reshape(w1, [self.env_data.M, (self.env_data.K + self.env_data.M)])
        w2 = np.reshape(w2, [self.env_data.M, (self.env_data.K + self.env_data.M)])
        #w = w + w1 + 1j*w2

        for m in range(self.env_data.M):
            for k in range(self.env_data.M + self.env_data.K):
                w[m, k] = w1[m, k] * (self.e ** (2*self.pi*1j*w2[m, k]))


        start = end
        end = end + self.env_data.N
        phi = (action[start: end]) * (2*self.pi)

        return w, phi

    def reset(self):
        s = np.zeros(self.state_size)
        return s.reshape(-1)

    def step(self, w, phi, h_dc, h_rc, h_dr, h_rr, G, snr, snr_max, action):
        done = False

        theta_1 = self.gen_Theta(phi)

        gamma_1 = self.cal_gamma_c(theta_1, w, G, h_dc, h_rc)
        R_1 = self.cal_R(gamma_1)
        #snr = self.cal_snr_r(theta, w, G, h_dr, h_rr, )
        EE_1 = self.EE(R_1, w)

        com_w = 0
        check_w = 0
        for k in range(self.env_data.K + self.env_data.M):
            com_w += (np.linalg.norm(w[:, k])) ** 2
        if com_w <= self.env_data.pmax:
            check_w = 1

        elif ((com_w) / 2) <= self.env_data.pmax:
            check_w = 2
        else:
            check_w = 0

        check_snr = 0

        next_state = self.state_cal(R_1, snr, h_dc, h_rc, h_dr, h_rr, G, snr_max, action)

        if snr >= self.env_data.snr:
            check_snr = 1

        elif 10 * snr >= self.env_data.snr:
            check_snr = 10
        elif 100 * snr >= self.env_data.snr:
            check_snr = 100
        elif 1000 * snr >= self.env_data.snr:
            check_snr = 1000
        else:
            check_snr = 0




        if check_snr == 1:
            if check_w == 1:
                reward = (100 * (np.sum(R_1)))
                done = True
            else:
                reward = self.env_data.pmax - com_w
        else:
            reward = -50

        #if check_w == 1:
            #if check_snr == 1:
                #reward = 1000 * np.sum(R_1)
                #done = True
            #elif check_snr == 10:
                #reward = 10 * np.sum(R_1)
            #elif check_snr == 100:
                #reward = -5
            #else:
                #reward = -10
        #elif check_w == 2:
         #   if check_snr == 1:
          #      reward = 10 * np.sum(R_1)
           # elif check_snr == 10:
            #    reward = -5
            #else:
             #   reward = -10
        #else:
         #   reward = -15



        return next_state, reward, done, check_snr, check_w, com_w, R_1, EE_1



































