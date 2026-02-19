import numpy as np
import dataloader as dl
import matlab.engine


class Env():
    def __init__(self, data):
        self.data = data
        self.result = data
        self.d1p = 25
        self.d2p = 10
        self.d3p = 25
        self.eng = matlab.engine.start_matlab()
        self.eng.cd('/Users/caglarcetinkaya/Documents/MATLAB/TMM - ML', nargout=0)
        self.avt_value, self.cri_value = self._get_location(self.d1p, self.d2p, self.d3p)
        self.current_step = 0
        self.max_steps = 500
        self.action_space = self._generate_action_space()
        self.n_actions = len(self.action_space)
        self.highest_avt = self.avt_value
        self.highest_cri = self.cri_value

    def _generate_action_space(self):
        actions = ["forward", "backward", "hold"]
        action_space = []
        for action1 in actions:
            for action2 in actions:
                for action3 in actions:
                    action_space.append([action1, action2, action3])
        return action_space
    """
    def _get_location(self, d1, d2, d3):
        location = self.result[(self.result['d1p'] == d1) & (self.result['d2p'] == d2) & (self.result['d3p'] == d3)]
        if location.empty:
            raise ValueError(f"Invalid position: d1={d1}, d2={d2}, d3={d3}")
        return location['AVT'].values[0], location['CRI_ext'].values[0]
    """
    def _get_location(self, d1p, d2p, d3p):
        avt_value, cri_value = self.eng.calculationTMMforPython(d1p, d2p, d3p, nargout=2)
        return avt_value, cri_value
    
    def reset(self):
        self.d1p = np.random.randint(1, 51) 
        self.d2p = np.random.randint(1, 21)
        self.d3p = np.random.randint(1, 51)
        self.current_step = 0
        return self._get_state()

    def step(self, action_idx):
        self.current_step += 1

        # Seçilen aksiyonları al
        action1, action2, action3 = self.action_space[action_idx]
        self.previous_avt = self.avt_value
        self.previous_cri = self.cri_value

        # d1 yönünde hareket
        if action1 == "forward" and self.d1p < 50:
            self.d1p += 1
        elif action1 == "backward" and self.d1p > 1:
            self.d1p -= 1
        elif action1 == "hold":
            pass

        # d2 yönünde hareket
        if action2 == "forward" and self.d2p < 20:
            self.d2p += 1
        elif action2 == "backward" and self.d2p > 1:
            self.d2p -= 1
        elif action2 == "hold":
            pass 

        # d3 yönünde hareket
        if action3 == "forward" and self.d3p < 50:
            self.d3p += 1
        elif action3 == "backward" and self.d3p > 1:
            self.d3p -= 1
        elif action3 == "hold":
            pass

        # Yeni AVT ve CRI değerlerini hesapla
        avt_value, cri_value = self._get_location(self.d1p, self.d2p, self.d3p)
        self.avt_value = avt_value
        self.cri_value = cri_value

        # Reward hesaplama
        alpha = 0.5  
        beta = 0.5   
        bonus_reward = 20
        penalty = 10

        reward = (
            alpha * (self.avt_value - self.previous_avt) +
            beta * (self.cri_value - self.previous_cri)
        )

        # Her iki metrikte de iyileşme varsa bonus
        if self.avt_value > self.previous_avt and self.cri_value > self.previous_cri:
            reward += bonus_reward
        # Bir metrik iyileşirken diğerinde düşüş varsa ceza
        elif self.avt_value > self.previous_avt and self.cri_value < self.previous_cri:
            reward -= penalty
        elif self.avt_value < self.previous_avt and self.cri_value > self.previous_cri:
            reward -= penalty

        done = self.current_step >= self.max_steps

        return self._get_state(), reward, done





    def _get_state(self):
        # Mevcut konumu döndür
        state = np.array([self.d1p, self.d2p, self.d3p], dtype=np.float32)
        return state

    def render(self):
        print(f"Current position: (d1={self.d1p}, d2={self.d2p}, d3={self.d3p}), Reflectance: {self.result[(self.result['d1p'] == self.d1p) & (self.result['d2p'] == self.d2p) & (self.result['d3p'] == self.d3p)]['AVT'].values[0]}, CRI: {self.result[(self.result['d1p'] == self.d1p) & (self.result['d2p'] == self.d2p) & (self.result['d3p'] == self.d3p)]['CRI_ext'].values[0]}")