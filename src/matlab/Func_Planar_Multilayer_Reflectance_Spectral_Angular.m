
function Ans = Func_Planar_Multilayer_Reflectance_Spectral_Angular(thicknesses,layers,angle)
% Set layers and thicknesses
% INPUTS
%layers = {'Vac' 'MgF2' 'ZnSe' 'InP' 'InGaAs' 'InP' 'Au'}; 
% Current list of available materials: MgF2, ZnSe, InGaAs, Au, InP
%thicknesses = [0 0 0 0.36 0.20 1.35 0]; % micron, note 0 means semi-infinite 0.192 0.107
%angle = 30;
% Are any of the layers thick films? (i.e. >> wavelength range of interest)
thick_film_index = [1];
wavelength_range = [1e-9 500e-9];

num_pts = 500;

% Wavelength space 
wL = linspace(min(wavelength_range),max(wavelength_range),num_pts); 

% lam = wavelength_range*1e6;
% lam_g = h*c./(Eg*1e-4); % microns
% lam_ag = lam(lam<=lam_g); % microns
% lam_bg = lam(lam>lam_g); % microns
% E_ag = h*c./(lam_ag*1e-4);
% E_bg = h*c./(lam_bg*1e-4);
% wL = [lam_ag lam_bg];  % wavelength in micron

t = thicknesses*1e-9; % in m
n = length(layers);

addpath(genpath([cd '/Materials/']));
N = zeros(length(wL),n);

for j = 1:n
    
    if strcmp(layers{j},'Vac')
        N(:,j) = 1;
    elseif strcmp(layers{j},'InGaAs')
        x = 0.47;
        layer_props = str2func(layers{j});
        N(:,j) = layer_props(wL,x);    
    else
        layer_props = str2func(layers{j});
        N(:,j) = layer_props(wL);
    end
    
end

% Thick Film Options
if ~isempty(thick_film_index)
    
Composite_Layer_1 =  1:thick_film_index;
Composite_Layer_2 =  thick_film_index:n;

R12 = Thin_Film_Reflectance(wL, t(Composite_Layer_1), N(:,Composite_Layer_1),angle);
R23 = Thin_Film_Reflectance(wL, t(Composite_Layer_2), N(:,Composite_Layer_2),angle);

alpha = (4*pi)./wL'.*imag(N(:,thick_film_index));
R_tot = R12 + (1-R12).^2.*R23.*exp(-2.*alpha.*t(thick_film_index)./cos(angle.*pi()/180))./(1-R12.*R23.*exp(-2.*alpha.*t(thick_film_index)./cos(angle.*pi()/180)));


plot(1./wL*1e-2,[R12 R23])
hold all,

else
    
R_tot = Thin_Film_Reflectance(wL, t, N, angle);

end


plot(1./wL*1e-2,[R_tot])
xlabel('Wavenumber (1/cm)');
ylabel('Reflectance');
ylim([0 1]);
xlim([0 1e4]);
hold all;






end



