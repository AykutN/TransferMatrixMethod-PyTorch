import torch
import numpy as np
import os
import sys
from scipy.interpolate import interp1d

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import config.settings as config

class TMMTorch:
    def __init__(self, device='cpu'):
        self.device = device
        self.matlab_src_dir = os.path.join(config.MATLAB_DIR)
        self.materials_dir = os.path.join(self.matlab_src_dir, 'Materials')
        self.properties_dir = os.path.join(self.materials_dir, 'Properties')
        
        # Physics Constants
        self.wavelength_min = 0.2  # microns
        self.wavelength_max = 1.2  # microns
        self.pts = 5001
        self.wL = torch.linspace(self.wavelength_min, self.wavelength_max, self.pts, dtype=torch.float64, device=self.device) * 1e-6
        self.wL_micron = self.wL * 1e6
        
        # Load Spectral References (A, B)
        self._load_references()
        
        # Preload Materials (Cache)
        self.materials_cache = {}
        
    def _load_references(self):
        # Load AM1.5G, RHE, CMFs, D65
        def load_prop(name, cols=(0, 1)):
            path = os.path.join(self.properties_dir, f"{name}.txt")
            data = np.loadtxt(path)
            # Interpolate to self.wL_micron
            # MATLAB produces NaNs outside range, effectively 0 when used with isnan checks later.
            # So we fill with 0.0 directly.
            f = interp1d(data[:, 0], data[:, cols[1]], kind='linear', fill_value=0.0, bounds_error=False)
            return torch.tensor(f(self.wL_micron.cpu().numpy()), dtype=torch.float64, device=self.device)

        self.S_AM15G = load_prop("AM15G")
        self.V_RHE = load_prop("RHE")
        self.D65 = load_prop("D65")
        
        # CMF (Lambda, X, Y, Z)
        path = os.path.join(self.properties_dir, "CMF_Lambda-x-y-z.txt")
        data = np.loadtxt(path)
        f_x = interp1d(data[:, 0], data[:, 1], kind='linear', fill_value=0.0, bounds_error=False)
        f_y = interp1d(data[:, 0], data[:, 2], kind='linear', fill_value=0.0, bounds_error=False)
        f_z = interp1d(data[:, 0], data[:, 3], kind='linear', fill_value=0.0, bounds_error=False)
        
        wL_np = self.wL_micron.cpu().numpy()
        self.CMF_X = torch.tensor(f_x(wL_np), dtype=torch.float64, device=self.device)
        self.CMF_Y = torch.tensor(f_y(wL_np), dtype=torch.float64, device=self.device)
        self.CMF_Z = torch.tensor(f_z(wL_np), dtype=torch.float64, device=self.device)
        
        # Precompute constants for A
        # PS = S_AM15G * V_RHE
        self.PS = self.S_AM15G * self.V_RHE
        self.PS[torch.isnan(self.PS)] = 0
        
        # Integrate PS (Trapezoidal)
        self.Int_PS = torch.trapz(self.PS, self.wL * 1e9)
        
        # Load CRI Data (SforD, Test Color Samples)
        self._load_cri_data()

    def _load_cri_data(self):
        # 1. Load SforD (Daylight Basis: S0, S1, S2)
        # Format: [Wavelength, S0, S1, S2]
        path_s = os.path.join(self.properties_dir, "SforD.txt")
        data_s = np.loadtxt(path_s) # 5001 lines usually
        
        # Interpolate to self.wL_micron
        f_s0 = interp1d(data_s[:, 0], data_s[:, 1], kind='linear', fill_value=0.0, bounds_error=False)
        f_s1 = interp1d(data_s[:, 0], data_s[:, 2], kind='linear', fill_value=0.0, bounds_error=False)
        f_s2 = interp1d(data_s[:, 0], data_s[:, 3], kind='linear', fill_value=0.0, bounds_error=False)
        
        wL_np = self.wL_micron.cpu().numpy()
        self.S0 = torch.tensor(f_s0(wL_np), dtype=torch.float64, device=self.device)
        self.S1 = torch.tensor(f_s1(wL_np), dtype=torch.float64, device=self.device)
        self.S2 = torch.tensor(f_s2(wL_np), dtype=torch.float64, device=self.device)
        
        # 2. Load Test Color Samples (TCS 1-15)
        # Format: [Wavelength, TCS1, ..., TCS15]
        # range 0.38 - 0.78 um
        path_tcs = os.path.join(self.properties_dir, "Test_Color_Samples.txt")
        data_tcs = np.loadtxt(path_tcs)
        
        # We need TCS 1 to 14 for CRI_ext (Mean of 1-14)
        # But MATLAB calculates 15. Let's load all 15. 
        # Columns: 0=WL, 1=TCS1 ... 15=TCS15 (Warning: indices shift)
        # File cols: 0=WL, 1=TCS1, ... 15=TCS15? Let's check logic:
        # Matlab: TCS1 = col 2 (index 1 in 0-based Python).
        # We have 15 TCS.
        
        self.TCS = []
        for i in range(15):
            col_idx = i + 1
            f_tcs = interp1d(data_tcs[:, 0], data_tcs[:, col_idx], kind='linear', fill_value=0.0, bounds_error=False)
            tcs_tensor = torch.tensor(f_tcs(wL_np), dtype=torch.float64, device=self.device)
            self.TCS.append(tcs_tensor)
            
        self.TCS = torch.stack(self.TCS, dim=1) # Shape: (Num_WL, 15)

    def calculate_color_metrics(self, T, R=None):
        # T: Transmittance spectrum (Batch, Num_WL) or (Num_WL,)
        if T.ndim == 1:
            T = T.unsqueeze(0)
            
        # 1. Calculate x_cr, y_cr (using D65)
        # STX = D65 * T * X
        STX = self.D65 * T * self.CMF_X
        STY = self.D65 * T * self.CMF_Y
        STZ = self.D65 * T * self.CMF_Z
        
        # Integrate (trapz over last dim)
        XX = torch.trapz(STX, self.wL * 1e9, dim=-1)
        YY = torch.trapz(STY, self.wL * 1e9, dim=-1)
        ZZ = torch.trapz(STZ, self.wL * 1e9, dim=-1)
        
        Sum_XYZ = XX + YY + ZZ + 1e-12 # avoid div 0
        x_cr = XX / Sum_XYZ
        y_cr = YY / Sum_XYZ
        
        # 2. Calculate CCT (Correlated Color Temperature)
        # Using Direct Transmittance as light source (as per MATLAB logic)
        # TX = T * CMF_X
        TX = T * self.CMF_X
        TY = T * self.CMF_Y
        TZ = T * self.CMF_Z
        
        XX_s = torch.trapz(TX, self.wL * 1e9, dim=-1)
        YY_s = torch.trapz(TY, self.wL * 1e9, dim=-1)
        ZZ_s = torch.trapz(TZ, self.wL * 1e9, dim=-1)
        
        Sum_s = XX_s + YY_s + ZZ_s + 1e-12
        x_salt = XX_s / Sum_s
        y_salt = YY_s / Sum_s
        
        # McCamy's Formula
        xe, ye = 0.3320, 0.1858
        n_mccamy = (x_salt - xe) / (y_salt - ye + 1e-12)
        n2 = n_mccamy ** 2
        n3 = n_mccamy ** 3
        CCT = -449 * n3 + 3525 * n2 - 6823.3 * n_mccamy + 5520.33
        
        # 3. Generate Reference Spectrum S_ref based on CCT
        # We need to handle batch CCT.
        # Logic:
        # if CCT <= 5100: Planckian
        # else: Daylight (S0 + M1*S1 + M2*S2)
        
        # Planckian
        # S_ref_planck = (2*pi*h*c^2 / lam^5) / (exp(hc/k*Tc*lam) - 1)
        # We compute this for all, then select.
        
        h_const = 6.6260694e-34
        c_const = 299792458
        k_const = 1.3806565e-23
        c2 = h_const * c_const / k_const # m K
        c1 = 2 * np.pi * h_const * c_const**2 # W m^2
        
        # Reshape for broadcasting
        # wL shape: (Num_WL)
        # CCT shape: (Batch)
        wL_m = self.wL.unsqueeze(0) # (1, 5001)
        CCT_exp = CCT.unsqueeze(1)  # (Batch, 1)
        
        # Plank Formula
        # exponent = c2 / (wL * T)
        exponent = c2 / (wL_m * CCT_exp)
        # Avoid overflow in exp
        exponent = torch.clamp(exponent, max=80.0) 
        
        planck = (c1 * (wL_m ** -5)) / (torch.exp(exponent) - 1.0)
        # Normalize to max 1 per batch
        planck_max = torch.max(planck, dim=1, keepdim=True)[0]
        S_ref_planck = planck / (planck_max + 1e-12)
        
        # Daylight Phase
        # if 5000 < CCT <= 7000:
        # kd = ...
        # But MATLAB just uses CCT directly in formulas.
        # Formulas match for > 5000 and <= 7000 vs > 7000.
        # Actually MATLAB splits logic.
        # But typically XD = ...
        # Let's use the formula from MATLAB exactly.
        
        # kd calculation
        # If <= 7000:
        kd_low = -4.6070e9 * (CCT**-3) + 2.9678e6 * (CCT**-2) + 0.09911e3 * (CCT**-1) + 0.244063
        # If > 7000:
        kd_high = -2.0064e9 * (CCT**-3) + 1.9018e6 * (CCT**-2) + 0.24748e3 * (CCT**-1) + 0.237040
        
        kd = torch.where(CCT <= 7000, kd_low, kd_high)
        
        ld = -3.000 * (kd**2) + 2.870 * kd - 0.275
        M1 = (-1.3515 - 1.7703*kd + 5.9114*ld) / (0.0241 + 0.2562*kd - 0.7341*ld)
        M2 = (0.0300 - 31.4424*kd + 30.0717*ld) / (0.0241 + 0.2562*kd - 0.7341*ld)
        
        # Combine S0, S1, S2
        # S_d = S0 + M1*S1 + M2*S2
        # S0: (Num_WL)
        # M1: (Batch)
        S_ref_day = self.S0.unsqueeze(0) + M1.unsqueeze(1) * self.S1.unsqueeze(0) + M2.unsqueeze(1) * self.S2.unsqueeze(0)
        
        # Normalize
        day_max = torch.max(S_ref_day, dim=1, keepdim=True)[0]
        S_ref_day = S_ref_day / (day_max + 1e-12)
        
        # Select S_ref
        mask_planck = (CCT <= 5100).unsqueeze(1)
        S_ref = torch.where(mask_planck, S_ref_planck, S_ref_day)
        
        # 4. CRI Calculation
        # We need Reference UVW and Test UVW for each of 14 TCS samples.
        # This can be vectorized over samples (Last dim = 14 samples, calculate all at once?)
        # Or loop. Since it's only 14, loop is fine, but vectorization is better for PyTorch.
        # Let's use the Tensor TCS (Num_WL, 15). We need 14.
        TCS14 = self.TCS[:, 0:14] # (Num_WL, 14)
        
        # Reference Source Values (Reference Illuminant on Reference Sample)
        # S_ref: (Batch, Num_WL)
        # TCS: (Num_WL, 14)
        # Output: (Batch, Num_WL, 14)
        
        S_ref_exp = S_ref.unsqueeze(2) # (B, W, 1)
        TCS_exp = TCS14.unsqueeze(0)   # (1, W, 14)
        CMF_X_exp = self.CMF_X.view(1, -1, 1) # (1, W, 1)
        CMF_Y_exp = self.CMF_Y.view(1, -1, 1)
        CMF_Z_exp = self.CMF_Z.view(1, -1, 1)
        
        # Integration function
        def integrate(tensor): return torch.trapz(tensor, self.wL * 1e9, dim=1) # collapses W dimension
        
        # Reference White Point (u_ref, v_ref)
        # Just S_ref on CMF
        XX_ref_src = integrate(S_ref * self.CMF_X) # (B)
        YY_ref_src = integrate(S_ref * self.CMF_Y)
        ZZ_ref_src = integrate(S_ref * self.CMF_Z)
        
        U_ref_src = (2/3) * XX_ref_src
        V_ref_src = YY_ref_src
        W_ref_src = 0.5 * (-XX_ref_src + 3*YY_ref_src + ZZ_ref_src)
        den_ref = U_ref_src + V_ref_src + W_ref_src + 1e-12
        u_ref = U_ref_src / den_ref
        v_ref = V_ref_src / den_ref
        
        # Von Kries c_r, d_r (Reference)
        c_r = (1/v_ref) * (4 - u_ref - 10*v_ref)
        d_r = (1/v_ref) * (1.708*v_ref + 0.404 - 1.481*u_ref)
        
        # Test Source White Point (u_k, v_k) - here 'k' means test source (transmitted light)
        # Calculated from T * CMF directly (as per Matlab x_salt calc)
        TX = T * self.CMF_X
        TY = T * self.CMF_Y
        TZ = T * self.CMF_Z
        
        XX_k_src = integrate(TX) # (B)
        YY_k_src = integrate(TY)
        ZZ_k_src = integrate(TZ)
        
        U_k_src = (2/3) * XX_k_src
        V_k_src = YY_k_src
        W_k_src = 0.5 * (-XX_k_src + 3*YY_k_src + ZZ_k_src)
        den_k = U_k_src + V_k_src + W_k_src + 1e-12
        u_k = U_k_src / den_k
        v_k = V_k_src / den_k
        
        # TCS Reference Values (S_ref * TCS)
        # Shape: (B, 14)
        SRS_X = integrate(S_ref_exp * CMF_X_exp * TCS_exp)
        SRS_Y = integrate(S_ref_exp * CMF_Y_exp * TCS_exp)
        SRS_Z = integrate(S_ref_exp * CMF_Z_exp * TCS_exp)
        
        U_ref_i = (2/3) * SRS_X
        W_ref_i = 0.5 * (-SRS_X + 3*SRS_Y + SRS_Z)
        den_ref_i = U_ref_i + SRS_Y + W_ref_i + 1e-12
        u_ref_i = U_ref_i / den_ref_i
        v_ref_i = SRS_Y / den_ref_i
        
        # TCS Test Values (T * TCS) - Note: T acts as Source Spectrum here
        # T: (B, W) -> (B, W, 1)
        T_exp = T.unsqueeze(2)
        TS_X = integrate(T_exp * CMF_X_exp * TCS_exp)
        TS_Y = integrate(T_exp * CMF_Y_exp * TCS_exp)
        TS_Z = integrate(T_exp * CMF_Z_exp * TCS_exp)
        
        U_k_i = (2/3) * TS_X
        W_k_i = 0.5 * (-TS_X + 3*TS_Y + TS_Z)
        den_k_i = U_k_i + TS_Y + W_k_i + 1e-12
        u_k_i = U_k_i / den_k_i
        v_k_i = TS_Y / den_k_i
        
        # Adaptive Shift (Von Kries)
        # c_ki, d_ki for each sample i under Test Source
        c_ki = (1 / v_k_i) * (4 - u_k_i - 10*v_k_i)
        d_ki = (1 / v_k_i) * (1.708*v_k_i + 0.404 - 1.481*u_k_i)
        
        # Expand scalars c_r, d_r, c_k, d_k etc to (B, 1) or (B, 14) as needed
        # u_ref, v_ref, u_k, v_k are (B)
        # Flattening logic for clarity:
        # All shapes should verify against (B, 14)
        
        def exp(t): return t.unsqueeze(1) # (B) -> (B, 1)
        
        c_r_e = exp(c_r)
        d_r_e = exp(d_r)
        # Note: Matlab uses 'c_k' and 'd_k' from the Source White Point
        # Recalculate c_k, d_k for source
        c_k_src = (1/v_k) * (4 - u_k - 10*v_k)
        d_k_src = (1/v_k) * (1.708*v_k + 0.404 - 1.481*u_k)
        c_k_e = exp(c_k_src)
        d_k_e = exp(d_k_src)
        
        # Shifted u', v' (u_ki_prime, v_ki_prime)
        # formula: uk' = (10.872 + 0.404 * (cr/ck)*cki - 4*(dr/dk)*dki) / (...)
        ratio_c = c_r_e / c_k_e
        ratio_d = d_r_e / d_k_e
        
        num_u = 10.872 + 0.404 * ratio_c * c_ki - 4 * ratio_d * d_ki
        den_uv = 16.518 + 1.481 * ratio_c * c_ki - ratio_d * d_ki
        num_v = 5.520
        
        u_ki_p = num_u / den_uv
        v_ki_p = num_v / den_uv
        
        # Lightness W
        # W_ri = 25 * (Yi_ref^(1/3)) - 17. Yi_ref is normalized? 100/YY_ref
        # Matlab: Wr1 = 25 * (((100/YY_ref)*YY_ref1)^(1/3)) - 17
        # YY_ref is source Y. YY_ref1 is sample Y.
        
        # Need to ensure YY_ref_src and YY_k_src are correctly shaped (B) -> (B, 1)
        YY_ref_src_e = exp(YY_ref_src)
        YY_k_src_e = exp(YY_k_src)
        
        W_ri = 25 * ( (100 * SRS_Y / YY_ref_src_e)**(1/3) ) - 17
        W_ki = 25 * ( (100 * TS_Y / YY_k_src_e)**(1/3) ) - 17
        
        # UVW Components
        # Ur_i = 13 * W_ri * (u_ref_i - u_ref)
        # Vr_i = 13 * W_ri * (v_ref_i - v_ref)
        U_ri = 13 * W_ri * (u_ref_i - exp(u_ref))
        V_ri = 13 * W_ri * (v_ref_i - exp(v_ref))
        
        # Uk_i = 13 * W_ki * (u_ki_p - u_k_p) ??
        # Matlab: Uk1 = 13 * Wk1 * (uk1_ - uk_)
        # uk_ is u_ref (Reference white point). Matlab line 316. 
        # This is because we transform the test sample to reference adaptability state.
        # So we subtract u_ref (Reference Source White), NOT Test Source White.
        U_ki = 13 * W_ki * (u_ki_p - exp(u_ref))
        V_ki = 13 * W_ki * (v_ki_p - exp(v_ref))
        
        # Delta E
        # Del_E1 = sqrt( (Ur1 - Uk1)^2 + (Vr1 - Vk1)^2 + (Wr1 - Wk1)^2 )
        dU = U_ri - U_ki
        dV = V_ri - V_ki
        dW = W_ri - W_ki
        Del_E = torch.sqrt(dU**2 + dV**2 + dW**2)
        
        # Ri = 100 - 4.6 * Del_E
        Ri = 100 - 4.6 * Del_E
        
        # CRI = mean(Ri)
        # CRI_ext is mean of 1..14
        CRI_ext = torch.mean(Ri, dim=1) # (B)
        
        return CRI_ext, x_cr, y_cr


    def load_material(self, name):
        if name in self.materials_cache:
            return self.materials_cache[name]
            
        if name == 'Vac':
            return torch.ones_like(self.wL, dtype=torch.complex128, device=self.device)

        p1 = os.path.join(self.materials_dir, f"{name}.txt")
        p2 = os.path.join(self.properties_dir, f"{name}.txt")
        
        path = None
        if os.path.exists(p1): path = p1
        elif os.path.exists(p2): path = p2
        
        if path:
            data = np.loadtxt(path)
            # Interpolate n and k
            # Format usually: [Wavelength, n, k] or [Wavelength, n]
            if data.shape[1] >= 3:
                f_n = interp1d(data[:, 0], data[:, 1], kind='linear', fill_value="extrapolate")
                f_k = interp1d(data[:, 0], data[:, 2], kind='linear', fill_value="extrapolate")
            else:
                 f_n = interp1d(data[:, 0], data[:, 1], kind='linear', fill_value="extrapolate")
                 f_k = lambda x: np.zeros_like(x)
                 
            wL_np = self.wL_micron.cpu().numpy()
            n = torch.tensor(f_n(wL_np), dtype=torch.float64, device=self.device)
            k = torch.tensor(f_k(wL_np), dtype=torch.float64, device=self.device)
            N = n + 1j * k
            self.materials_cache[name] = N
            return N
        else:
            raise FileNotFoundError(f"Material {name} not found in {self.materials_dir} or {self.properties_dir}")

    def materials_cache_dir(self, name):
         # Just a helper to point to potential locations
         return ""

    def tmm_transfer(self, layers, thicknesses, angle=0):
        # layers: list of complex tensors (N) corresponding to each layer
        # thicknesses: list of tensors (d) for each layer (in METERS)
        # angle: incidence angle in radians
        
        c = 2.9979e8
        wL = self.wL
        
        num_layers = len(layers)
        batch_size = layers[0].shape[0] # Usually = number of wavelengths
        
        # Incident Angle propagation (Snell's Law equivalent for complex media)
        # o[j] = asin( N[0]/N[j] * sin(o[0]) )
        # Note: N is complex, so o is complex.
        
        o = [None] * num_layers
        o[0] = torch.tensor(angle, dtype=torch.complex128, device=self.device).expand(batch_size) # Scalar to tensor
        
        # Calculate angles in all layers
        # Watch out for complex asin. PyTorch supports it.
        for j in range(1, num_layers):
            sin_oj = (layers[0] / layers[j]) * torch.sin(o[0])
            o[j] = torch.asin(sin_oj)
            
        # P's and M's
        p_p = [None] * num_layers
        p_s = [None] * num_layers
        
        for j in range(num_layers):
            cos_oj = torch.cos(o[j])
            p_p[j] = cos_oj / (layers[j] / c)
            p_s[j] = cos_oj / (-c / layers[j])
            
        # Initialize M matrices
        # M is 2x2. For vectorized wL, it's (Batch, 2, 2)
        M_p = torch.eye(2, dtype=torch.complex128, device=self.device).unsqueeze(0).expand(batch_size, 2, 2).clone()
        M_s = torch.eye(2, dtype=torch.complex128, device=self.device).unsqueeze(0).expand(batch_size, 2, 2).clone()
        
        # Iterate over INTERFACES and PULSES (Matrices represent layers)
        # Loop from layer 1 (index 1) to n-2 ? 
        # MATLAB code logic:
        # for j=2:(n-1)
        #   calculate phase phi(j)
        #   create M_pj, M_sj
        #   multiply
        
        # MATLAB indices are 1-based. Python 0-based.
        # Layers in MATLAB: 1=Vac, 2=MoO3... n=Vac
        # Loop j=2 to n-1 means INNER layers (not first/last semi-infinite media).
        
        for j in range(1, num_layers - 1):
             d_j = thicknesses[j] # thickness of layer j
             # phi = 2*pi*d*N*cos(o)/lambda
             phi_j = 2 * np.pi * d_j * layers[j] * torch.cos(o[j]) / wL
             
             cos_phi = torch.cos(phi_j)
             sin_phi = torch.sin(phi_j)
             
             # M_pj construction
             # [cos -i*pp*sin]
             # [-i/pp*sin cos]
             
             # We need to construct (Batch, 2, 2)
             # Elements:
             m11 = cos_phi
             m12 = -1j * p_p[j] * sin_phi
             m21 = -1j / p_p[j] * sin_phi
             m22 = cos_phi
             
             # Stack them
             M_pj = torch.stack([torch.stack([m11, m12], dim=-1), 
                                 torch.stack([m21, m22], dim=-1)], dim=-2)
             
             # M_sj
             m11s = cos_phi
             m12s = -1j * p_s[j] * sin_phi
             m21s = -1j / p_s[j] * sin_phi
             m22s = cos_phi
             
             M_sj = torch.stack([torch.stack([m11s, m12s], dim=-1), 
                                 torch.stack([m21s, m22s], dim=-1)], dim=-2)
             
             M_p = torch.bmm(M_p, M_pj)
             M_s = torch.bmm(M_s, M_sj)
             
        # Coefficients
        # n is last index
        n = num_layers - 1
        
        # p or TM
        # r_p = ((M11 + M12/p_n) - (M21 + M22/p_n)*p_1) / (...)
        # p_1 is p_p[0], p_n is p_p[n]
        
        m11 = M_p[:, 0, 0]
        m12 = M_p[:, 0, 1]
        m21 = M_p[:, 1, 0]
        m22 = M_p[:, 1, 1]
        
        pp0 = p_p[0]
        ppn = p_p[n]
        
        term1 = m11 + (1./ppn)*m12
        term2 = (m21 + (1./ppn)*m22) * pp0
        
        r_p = (term1 - term2) / (term1 + term2)
        Rp = r_p * torch.conj(r_p)
        
        # Transmissivity
        c_p = torch.cos(o[0]) / torch.cos(o[n])
        t_p = 2 * c_p / (term1 + term2)
        Tp = torch.real(torch.conj(layers[n]) * torch.cos(o[n])) / \
             torch.real(torch.conj(layers[0]) * torch.cos(o[0])) * (t_p * torch.conj(t_p))
             
        # s or TE
        m11s = M_s[:, 0, 0]
        m12s = M_s[:, 0, 1]
        m21s = M_s[:, 1, 0]
        m22s = M_s[:, 1, 1]
        
        ps0 = p_s[0]
        psn = p_s[n]
        
        term1s = m11s + (1./psn)*m12s
        term2s = (m21s + (1./psn)*m22s) * ps0
        
        r_s = (term1s - term2s) / (term1s + term2s)
        Rs = r_s * torch.conj(r_s)
        
        t_s = 2 / (term1s + term2s) * (layers[0]/layers[n]) * (torch.cos(o[0])/torch.cos(o[n]))
        
        # Alternative Ts formula in MATLAB check:
        # t_s = 2 / ((M_s(1,1) + 1./p_s(n).*M_s(1,2))+(M_s(2,1) + 1./p_s(n).*M_s(2,2)).*p_s(1))*N(w,1)/N(w,n)*cos(o(1))/cos(o(n));
        # YES.
        
        Ts = torch.real(layers[n] * torch.cos(o[n])) / \
             torch.real(layers[0] * torch.cos(o[0])) * (t_s * torch.conj(t_s))
             
             
        R_tot = 0.5 * (torch.real(Rp) + torch.real(Rs))
        T_tot = 0.5 * (torch.real(Tp) + torch.real(Ts))
        
        return R_tot, T_tot

    def forward(self, *d_args):
        # We allow d_args to be individual arguments or a single list/tuple
        if len(d_args) == 1 and isinstance(d_args[0], (list, tuple, torch.Tensor)):
            d_args = d_args[0]
            
        # Validate count
        current_layers = config.LAYERS
        if len(d_args) != len(current_layers):
            # Fallback for old code calling with 6 args explicitly if config changed
            # But user wants flexibility, so better to warn or error.
            # For now, let's just proceed with strict check.
            pass 
            
        # Materials (Vac + Defined Stack + Vac)
        names = ['Vac'] + list(current_layers) + ['Vac']
        layer_mats = [self.load_material(n) for n in names]
        
        # Thicknesses (convert from nm to m)
        def to_m(d): return d * 1e-9
        
        # Construct thickness list: 1nm (Vac) + d_args + 1nm (Vac)
        # Handle tensor vs float inputs
        
        thicks = [to_m(torch.tensor(1.0, device=self.device))]
        
        for d in d_args:
            if isinstance(d, (int, float)):
                thicks.append(to_m(torch.tensor(d, device=self.device)))
            else:
                thicks.append(to_m(d))
                
        thicks.append(to_m(torch.tensor(1.0, device=self.device)))
        
        R, T = self.tmm_transfer(layer_mats, thicks, angle=0)
        
        # A
        # PST = PS .* T_tot
        PST = self.PS * T
        PST[torch.isnan(PST)] = 0
        Int_PST = torch.trapz(PST, self.wL * 1e9)
        AVT = (Int_PST / self.Int_PS) * 100
        
        # Calculate Color Metrics (CRI, x, y)
        CRI_ext, x_cr, y_cr = self.calculate_color_metrics(T)
        
        # B (Short Circuit Current)
        # Using exact formula from calculationTMMforPython.m:
        # B = (1/(h*c))*trapz(wL*1E9, A_tot.*S_AM15G)*100*1E-9
        
        A_tot = 1.0 - R - T
        B_integrant = A_tot * self.S_AM15G
        B_integrant[torch.isnan(B_integrant)] = 0.0
        
        # Constants from MATLAB code
        h = 4.13566766225e-15 # eV s
        c = 2.9979e8          # m/s
        
        # Integral
        # trapz requires explicit x if dx is not 1. wL is non-uniform? No, linspace.
        # But let's be safe and use x=self.wL*1e9
        integral = torch.trapz(B_integrant, self.wL * 1e9)
        
        Jph = (1/(h*c)) * integral * 100 * 1e-9
        
        return AVT, Jph, CRI_ext, x_cr, y_cr

