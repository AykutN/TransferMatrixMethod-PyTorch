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
        self.eng = matlab.engine.start_matlab()
        self.cache = {}
        self.eng.cd("/Users/caglarcetinkaya/Documents/MATLAB/TMM - ML", nargout=0)
        self.threshold = 5
        self.avt_value, self.jph_value = self._get_location(self.d1p, self.d2p, self.d3p, self.d4p, self.d5p, self.d6p, self.d7p)
        self.max_steps = 500
        self.action_space = self._generate_action_space()
        self.n_actions = len(self.action_space)
        self.highest_avt = self.avt_value
        self.highest_jph = self.jph_value

    def _generate_action_space(self):
        categories = ["d1", "d2", "d3", "d4", "d5", "d6", "d7"]  
        actions = ["forward", "backward"]  
        action_space = []
        
        for category in categories:
            for action in actions:
                action_space.append([category, action])  

        return action_space

        

    def _get_location(self, d1p, d2p, d3p, d4p, d5p, d6p, d7p):
        key = (d1p, d2p, d3p, d4p, d5p, d6p, d7p)
        if key not in self.cache:
            avt,jph = self.eng.calculationTMMforPython(float(d1p), float(d2p), float(d3p), float(d4p), float(d5p), float(d6p), float(d7p), nargout=2)
            self.cache[key] = (avt, jph)
        return self.cache[key]

    def reset(self):
        self.d1p = np.random.randint(5, 71) 
        self.d2p = np.random.randint(5, 26)
        self.d3p = np.random.randint(10, 51)
        self.d4p = np.random.randint(50, 151)
        self.d5p = np.random.randint(10, 51)
        self.d6p = np.random.randint(5, 26)
        self.d7p = np.random.randint(5, 71)
        self.current_step = 0
        return self._get_state()

    def step(self, action_idx):
        self.current_step += 1

        category, action = self.action_space[action_idx]

        self.previous_avt, self.previous_jph = self.avt_value, self.jph_value

        bounds = {
            "d1": (5, 70),
            "d2": (5, 25),
            "d3": (10, 50),
            "d4": (50, 150),
            "d5": (10, 50),
            "d6": (5, 25),
            "d7": (5, 70),
        }
        if category == "d1":
            if action == "forward":
                self.d1p += 2
            elif action == "backward" and self.d1p > 2:
                self.d1p -= 2
            self.d1p = max(bounds["d1"][0], min(self.d1p, bounds["d1"][1]))

        elif category == "d2":
            if action == "forward":
                self.d2p += 1
            elif action == "backward" and self.d2p > 1:
                self.d2p -= 1
            self.d2p = max(bounds["d2"][0], min(self.d2p, bounds["d2"][1]))

        elif category == "d3":
            if action == "forward":
                self.d3p += 1
            elif action == "backward" and self.d3p > 1:
                self.d3p -= 1
            self.d3p = max(bounds["d3"][0], min(self.d3p, bounds["d3"][1]))

        elif category == "d4":
            if action == "forward":
                self.d4p += 2
            elif action == "backward" and self.d4p > 2:
                self.d4p -= 2
            self.d4p = max(bounds["d4"][0], min(self.d4p, bounds["d4"][1]))

        elif category == "d5":
            if action == "forward":
                self.d5p += 1
            elif action == "backward" and self.d5p > 1:
                self.d5p -= 1
            self.d5p = max(bounds["d5"][0], min(self.d5p, bounds["d5"][1]))

        elif category == "d6":
            if action == "forward":
                self.d6p += 1
            elif action == "backward" and self.d6p > 1:
                self.d6p -= 1
            self.d6p = max(bounds["d6"][0], min(self.d6p, bounds["d6"][1]))

        elif category == "d7":
            if action == "forward":
                self.d7p += 2
            elif action == "backward" and self.d7p > 2:
                self.d7p -= 2
            self.d7p = max(bounds["d7"][0], min(self.d7p, bounds["d7"][1]))

        
        self.avt_value, self.jph_value = self._get_location(
            self.d1p, self.d2p, self.d3p, self.d4p, self.d5p, self.d6p, self.d7p
        )

        self.highest_avt = max(self.highest_avt, self.avt_value)
        self.highest_jph = max(self.highest_jph, self.jph_value)
        
        
        
        
        reward = 0

        # Jph'yi maksimize etmeye odaklanıyoruz
        if self.jph_value > self.highest_jph and self.avt_value >= 25:
            reward += 5  # Jph'nin en yüksek değere ulaştığı ve AVT'nin kabul edilebilir sınırlar içinde olduğu durumda büyük ödül
        elif self.jph_value > self.previous_jph and self.avt_value >= 25:
            reward += 2  # Jph artıyorsa ve AVT kabul edilebilir seviyedeyse ödül ver
        elif self.jph_value < self.previous_jph:
            reward -= 2  # Jph azalıyorsa ceza ver

        # AVT'nin alt sınırı kontrol ediliyor
        if self.avt_value < 25:
            reward -= 2  # AVT 25'in altına düşerse ceza ver
        elif 25 <= self.avt_value <= 100:
            reward += 1  # AVT kabul edilebilir aralıkta ise küçük bir ödül ver

        # AVT ve Jph'nin optimum dengesi için bir ek ödül mekanizması
        if self.jph_value > self.previous_jph and 25 <= self.avt_value <= 35:
            reward += 2  # Hem Jph artıyor hem de AVT 25-50 arasında ise ek ödül ver
        if self.jph_value > self.previous_jph and 25 <= self.avt_value <= 26:
            reward += 4  # Hem Jph artıyor hem de AVT 25-26 arasında ise ek ödül ver

        # Epizodun bitip bitmediğini kontrol et
        done = self.current_step >= self.max_steps

        return self._get_state(), reward, done
        
       



    def _get_state(self):
       
        state = np.array([self.d1p, self.d2p, self.d3p, self.d4p, self.d5p, self.d6p, self.d7p], dtype=np.float32)
        return state
    
    def render(self):
        print(f"Current position: (d1={self.d1p}, d2={self.d2p}, d3={self.d3p}, d4={self.d4p}, d5={self.d5p}, d6={self.d6p}, d7={self.d7p}), Reflectance: {self.result[(self.result['d1p'] == self.d1p) & (self.result['d2p'] == self.d2p) & (self.result['d3p'] == self.d3p) & (self.result['d4p'] == self.d4p) & (self.result['d5p'] == self.d5p) & (self.result['d6p'] == self.d6p) & (self.result['d7p'] == self.d7p)]['AVT'].values[0]}")

