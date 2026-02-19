import numpy as np

import matlab.engine

class Env():
    def __init__(self):
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
    """
    def _get_location(self, d1p, d2p, d3p, d4p, d5p, d6p, d7p):
        # Örnek bir simülasyon (gerçek formülünüzü ekleyin)
        avt = 30 - 0.1*d1p + 0.2*d2p - 0.05*d3p + 0.15*d4p - 0.07*d5p + 0.1*d6p - 0.05*d7p
        jph = 25 + 0.15*d1p - 0.1*d2p + 0.2*d3p - 0.05*d4p + 0.1*d5p - 0.07*d6p + 0.2*d7p
        avt = np.clip(avt, 0, 30)
        jph = np.clip(jph, 0, 50)
        return avt, jph
    """
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

        penalty = 0  

        if category == "d1":
            if action == "forward":
                self.d1p += 1
                if self.d1p > bounds["d1"][1]:
                    penalty -= 10 
            elif action == "backward":
                self.d1p -= 1
                if self.d1p < bounds["d1"][0]:
                    penalty -= 10
            self.d1p = max(bounds["d1"][0], min(self.d1p, bounds["d1"][1]))

        elif category == "d2":
            if action == "forward":
                self.d2p += 1
                if self.d2p > bounds["d2"][1]:
                    penalty -= 10
            elif action == "backward":
                self.d2p -= 1
                if self.d2p < bounds["d2"][0]:
                    penalty -= 10
            self.d2p = max(bounds["d2"][0], min(self.d2p, bounds["d2"][1]))

        elif category == "d3":
            if action == "forward":
                self.d3p += 1
                if self.d3p > bounds["d3"][1]:
                    penalty -= 10
            elif action == "backward":
                self.d3p -= 1
                if self.d3p < bounds["d3"][0]:
                    penalty -= 10
            self.d3p = max(bounds["d3"][0], min(self.d3p, bounds["d3"][1]))

        elif category == "d4":
            if action == "forward":
                self.d4p += 1
                if self.d4p > bounds["d4"][1]:
                    penalty -= 10
            elif action == "backward":
                self.d4p -= 1
                if self.d4p < bounds["d4"][0]:
                    penalty -= 10
            self.d4p = max(bounds["d4"][0], min(self.d4p, bounds["d4"][1]))

        elif category == "d5":
            if action == "forward":
                self.d5p += 1
                if self.d5p > bounds["d5"][1]:
                    penalty -= 10
            elif action == "backward":
                self.d5p -= 1
                if self.d5p < bounds["d5"][0]:
                    penalty -= 10
            self.d5p = max(bounds["d5"][0], min(self.d5p, bounds["d5"][1]))

        elif category == "d6":
            if action == "forward":
                self.d6p += 1
                if self.d6p > bounds["d6"][1]:
                    penalty -= 10
            elif action == "backward":
                self.d6p -= 1
                if self.d6p < bounds["d6"][0]:
                    penalty -= 10
            self.d6p = max(bounds["d6"][0], min(self.d6p, bounds["d6"][1]))

        elif category == "d7":
            if action == "forward":
                self.d7p += 1
                if self.d7p > bounds["d7"][1]:
                    penalty -= 10
            elif action == "backward":
                self.d7p -= 1
                if self.d7p < bounds["d7"][0]:
                    penalty -= 10
            self.d7p = max(bounds["d7"][0], min(self.d7p, bounds["d7"][1]))

        
        self.avt_value, self.jph_value = self._get_location(
            self.d1p, self.d2p, self.d3p, self.d4p, self.d5p, self.d6p, self.d7p
        )
        
        reward_vector = self.calculate_rewards()


        done = self.current_step >= self.max_steps

        return self._get_state(), reward_vector, done


    def _get_state(self):
       
        state = np.array([self.d1p, self.d2p, self.d3p, self.d4p, self.d5p, self.d6p, self.d7p], dtype=np.float32)
        return state

    def calculate_rewards(self):
        avt_target = 25
        avt_ideal_range = (25, 27)  # Değiştirildi: 25-27 arası ideal
        
        # AVT için ödül hesaplama
        if self.avt_value >= avt_ideal_range[0]:
            if avt_ideal_range[0] <= self.avt_value <= avt_ideal_range[1]:
                reward_avt = 150  # İdeal aralıkta maksimum ödül
            else:
                excess = self.avt_value - avt_ideal_range[1]
                reward_avt = 70 - (20 * excess)  # Aralık dışına çıktıkça azalan ödül
        else:
            distance = avt_ideal_range[0] - self.avt_value
            reward_avt = -50 * (distance / avt_target) ** 2  # AVT düşükse kuadratik ceza
        
        # JPH için ödül hesaplama
        jph_improvement = self.jph_value - self.previous_jph
        if jph_improvement > 0:
            reward_jph = 50 + (50 * jph_improvement)  # Daha yüksek ödül
        else:
            reward_jph = -10 * abs(jph_improvement)  # Daha düşük ceza
        
        # Stability bonus: AVT değeri ideal aralıkta kalıyorsa
        if (25 <= self.previous_avt <= 27) and (25 <= self.avt_value <= 27):
            reward_avt += 50
        
        # Multi-objective ağırlıklandırma
        avt_weight = 0.7 if self.avt_value < avt_target else 0.3
        jph_weight = 1 - avt_weight
        
        # Her iki hedef de iyiyse bonus
        if avt_ideal_range[0] <= self.avt_value <= avt_ideal_range[1] and jph_improvement > 0:
            reward_avt += 50
            reward_jph += 50
        
        reward_vector = [avt_weight * reward_avt, jph_weight * reward_jph]
        return reward_vector


