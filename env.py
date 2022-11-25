import random

import numpy as np
import copy

SLOT_TIME = 9
SIFS = 16
DIFS = 34
ACK = 18
DATA_SIZE = 1000
SEND_SPEED = 148
ONE_SECOND = 1000000

class Env():
    def __init__(self, st_num):
        self.cw_arr = [15, 31, 63, 127, 255, 511, 1023]
        self.station_backoff = np.array([random.randint(0, 15) for _ in range(st_num)])
        self.th_arr = [0 for _ in range(len(self.cw_arr))]

    def csma_ca(self, action):
        success_num = 0
        collision_num = 0
        step_time = 0

        while step_time < ONE_SECOND:
            backoff_min = np.min(self.station_backoff)
            # 패킷을 전송할 스태이션이 있을경우
            if backoff_min == 0:
                send = np.where(self.station_backoff == 0)[0]
                send = send.tolist()
                if len(send) == 1: # 전송성공
                    i = send[0]
                    success_num += 1
                    self.station_backoff[i] = np.random.randint(self.cw_arr[action])
                    step_time += SEND_SPEED + SIFS + ACK + DIFS
                else: # 충동 발생
                    for i in send:
                        collision_num += 1
                        self.station_backoff[i] = np.random.randint(self.cw_arr[action])
                    step_time += SEND_SPEED + SIFS + DIFS
            # 전송할 스태이션이 없을경우
            else:
                self.station_backoff -= 1
                step_time += SLOT_TIME

        throughput = self.get_throughput(success_num, step_time)
        reward = self.get_reward(throughput, action)

        return [throughput, self.cw_arr[action]], reward

    def get_throughput(self, success, step_time):
        return success * DATA_SIZE * 8.0 / step_time

    def get_reward(self, throughput, action):
        self.th_arr[action] = throughput

        if max(self.th_arr) == throughput:
            return 1.0
        return -1.0
