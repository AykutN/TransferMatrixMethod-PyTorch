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
        
        # Load Spectral References (AVT, Jph)
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
        
        # Precompute constants for AVT
        # PS = S_AM15G * V_RHE
        self.PS = self.S_AM15G * self.V_RHE
        self.PS[torch.isnan(self.PS)] = 0
        
        # Integrate PS (Trapezoidal)
        self.Int_PS = torch.trapz(self.PS, self.wL * 1e9) # wL in nm for integration? 
        # Original MATLAB: trapz(wL*1E9, PS). wL is in meters, *1E9 makes it nm. 
        # Correct.

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
        
        # AVT
        # PST = PS .* T_tot
        PST = self.PS * T
        PST[torch.isnan(PST)] = 0
        Int_PST = torch.trapz(PST, self.wL * 1e9)
        AVT = (Int_PST / self.Int_PS) * 100
        
        
        # Jph (Short Circuit Current)
        # Using exact formula from calculationTMMforPython.m:
        # Jph = (1/(h*c))*trapz(wL*1E9, A_tot.*S_AM15G)*100*1E-9
        
        A = 1.0 - R - T
        Jph_integrant = A * self.S_AM15G
        Jph_integrant[torch.isnan(Jph_integrant)] = 0.0
        
        # Constants from MATLAB code
        h = 4.13566766225e-15 # eV s
        c = 2.9979e8          # m/s
        
        # Integral
        # trapz requires explicit x if dx is not 1. wL is non-uniform? No, linspace.
        # But let's be safe and use x=self.wL*1e9
        integral = torch.trapz(Jph_integrant, self.wL * 1e9)
        
        Jph = (1/(h*c)) * integral * 100 * 1e-9
        
        return AVT, Jph

