import numpy as np
from dqn import Qnet
from buffer import ReplayBuffer
from env import Env
import torch


def simulation(node_num):
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    buffer = ReplayBuffer()

    throughputs = []  # 에피소드 throughput 저장

    env = Env(node_num)  # env 설정
    state, _ = env.csma_ca(0)  # 초기 CW 15
    for epi in range(100000):
        epsilon = max(0.01, 0.5 * (1 / (epi + 1)))  # 탐험 확률 조절
        action = q.get_action(torch.from_numpy(np.array(state)).float(), epsilon)
        state_prime, reward = env.csma_ca(action)

        buffer.put((state, action, reward, state_prime))
        state = state_prime
        throughputs.append(state[0])

        if buffer.size() > buffer.batch_size:
            q.train_net(q, q_target, buffer)
        # 20 epi마다 타겟 변경
        if epi % 20 == 0 and epi != 0:
            q_target.load_state_dict(q.state_dict())

    return np.mean(throughputs[-100:])


if __name__ == '__main__':
    throughputs = []
    node_nums = [(i + 1) * 10 for i in range(10)]

    for node in node_nums:
        throughput = simulation(node)
        throughputs.append(throughput)

    print(throughputs)
