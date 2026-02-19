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
        self.d4p = 150
        self.d5p = 30
        self.d6p = 30
        self.d7p = 40
        self.avt_value, self.cri_value = self._get_location(self.d1p, self.d2p, self.d3p, self.d4p, self.d5p, self.d6p, self.d7p)
        eng = matlab.engine.start_matlab()
        eng.cd('/Users/caglarcetinkaya/Documents/MATLAB/TMM - ML', nargout=0) 
        self.current_step = 0
        self.max_steps = 500
        self.action_space = self._generate_action_space()
        self.n_actions = len(self.action_space)
        self.highest_avt = self.avt_value

    def _generate_action_space(self):
        actions = ["forward", "backward", "hold"]
        action_space = []
        for action1 in actions:
            for action2 in actions:
                for action3 in actions:
                    for action4 in actions:
                        for action5 in actions:
                            for action6 in actions:
                                for action7 in actions:
                                    action_space.append([action1, action2, action3, action4, action5, action6, action7])

        return action_space
        
    def _get_location(self, d1p, d2p, d3p, d4p, d5p, d6p, d7p):
        eng = matlab.engine.start_matlab()
        avt_value = eng.calculationTMMforPython(d1p, d2p, d3p, d4p, d5p, d6p, d7p, nargout=2)
        return avt_value

    def reset(self):
        self.d1p = np.random.randint(1, 51) 
        self.d2p = np.random.randint(1, 21)
        self.d3p = np.random.randint(1, 51)
        self.d4p = np.random.randint(50, 251)
        self.d5p = np.random.randint(10, 51)
        self.d6p = np.random.randint(5, 51)
        self.d7p = np.random.randint(5, 71)
        self.current_step = 0
        return self._get_state()

    def step(self, action_idx):
        self.current_step += 1

        # Seçilen aksiyonları al
        action1, action2, action3, action4, action5, action6, action7 = self.action_space[action_idx]
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

        # d4 yönünde hareket
        if action4 == "forward" and self.d4p < 250:
            self.d4p += 1
        elif action4 == "backward" and self.d4p > 50:
            self.d4p -= 1
        elif action4 == "hold":
            pass

        # d5 yönünde hareket
        if action5 == "forward" and self.d5p < 51:
            self.d5p += 1
        elif action5 == "backward" and self.d5p > 10:
            self.d5p -= 1
        elif action5 == "hold":
            pass

        # d6 yönünde hareket
        if action6 == "forward" and self.d6p < 51:
            self.d6p += 1
        elif action6 == "backward" and self.d6p > 5:
            self.d6p -= 1
        elif action6 == "hold":
            pass

        # d7 yönünde hareket
        if action7 == "forward" and self.d7p < 71:
            self.d7p += 1
        elif action7 == "backward" and self.d7p > 5:
            self.d7p -= 1
        elif action7 == "hold":
            pass

        # Yeni AVT ve CRI değerlerini hesapla
        avt_value = self._get_location(self.d1p, self.d2p, self.d3p, self.d4p, self.d5p, self.d6p, self.d7p)
        self.avt_value = avt_value
        

        # Reward hesaplama

        reward = (
            (self.avt_value - self.previous_avt)
            #(self.cri_value - self.previous_cri)
        )

        if avt_value > self.previous_avt: 
            reward = 10
        elif avt_value < self.previous_avt:
            reward = -10


        done = self.current_step >= self.max_steps

        return self._get_state(), reward, done





    def _get_state(self):
        # Mevcut konumu döndür
        state = np.array([self.d1p, self.d2p, self.d3p, self.d4p, self.d5p, self.d6p, self.d7p], dtype=np.float32)
        return state
    
    def render(self):
        print(f"Current position: (d1={self.d1p}, d2={self.d2p}, d3={self.d3p}, d4={self.d4p}, d5={self.d5p}, d6={self.d6p}, d7={self.d7p}), Reflectance: {self.result[(self.result['d1p'] == self.d1p) & (self.result['d2p'] == self.d2p) & (self.result['d3p'] == self.d3p) & (self.result['d4p'] == self.d4p) & (self.result['d5p'] == self.d5p) & (self.result['d6p'] == self.d6p) & (self.result['d7p'] == self.d7p)]['AVT'].values[0]}")

env = Env(dl.data)
env.action_space()