function [wL,A_tot] = Func_Planar_Multilayer_Spectral_Angular(thicknesses,layers,angle)
% Set layers and thicknesses
%INPUT: ([layer thicknesses (microns)],{Material file names},angle of incidence)
%Example cell: 15deg incidence, Vacuum, 0.2micron MgF2, 0.1micron ZnSe, 1.3micron InGaAs, 0.4micron Au
%Example input: ([0 0.2 0.1 1.3 0],{'Vac','MgF2','ZnSe','InGaAs','Au'},15)
%*0 denotes semi-infinite geometry

% Are any of the layers thick films? (i.e. >> wavelength range of interest)
thick_film_index = [2];
wavelength_min = 0.35 ; % microns
wavelength_max = 3; % microns
pts = 1e3;

wL = linspace(wavelength_min,wavelength_max,pts)*1e-6;
wL = wL';

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
% N(:,3) = 1.5 + 5e-7i;
R = zeros(length(wL),length(angle));
T = zeros(length(wL),length(angle));
A = zeros(length(wL),length(angle));

for ii = 1:length(angle)
% Thick Film Options
if ~isempty(thick_film_index)
    
Composite_Layer_1 =  1:thick_film_index;
Composite_Layer_2 =  thick_film_index:n;

[R12s,R12p,T12s,T12p,angle_f] = TMM_Ref_Trans_Thick(wL, t(Composite_Layer_1), N(:,Composite_Layer_1),angle(ii));
[R23s,R23p,T23s,T23p] = TMM_Ref_Trans_Thick(wL, t(Composite_Layer_2), N(:,Composite_Layer_2),angle_f);

n_inc = N(:,1);
n_thick = N(:,thick_film_index);
k_thick = imag(N(:,thick_film_index));

alpha = (4*pi)./wL.*k_thick;
angle_film = (asin(n_inc./n_thick.*sin(angle(ii)*pi/180)));

path = t(thick_film_index)./cos(angle_film);

beta = alpha.*path;
phi = 2*pi*n_thick*t(thick_film_index).*cos(angle_film)./wL;
Rs = R12s + (T12s.^2.*R23s.*exp(-2.*beta))./(1-R12s.*R23s.*exp(-2.*beta));
Rp = R12p + (T12p.^2.*R23p.*exp(-2.*beta))./(1-R12p.*R23p.*exp(-2.*beta));
Ts = T12s.*T23s.*exp(-beta)./(1 - R12s.*R23s.*exp(-2*beta));
Tp = T12p.*T23p.*exp(-beta)./(1 - R12p.*R23p.*exp(-2*beta));

R_tot(:,ii) = (Rs + Rp)/2;
T_tot(:,ii) = (Ts + Tp)/2;
A_tot(:,ii) = 1 - R_tot - T_tot;


else
    
[R_tot,T_tot] = TMM_Ref_Trans(wL, t, N, angle(ii));
A_tot = 1 - R_tot - T_tot;

end
R(:,ii) = R_tot;
T(:,ii) = T_tot;
A(:,ii) = A_tot;
end
wL = wL*1e6;
figure(1)
plot(wL,R_tot,wL,T_tot,wL,A_tot)
xlabel('Wavelength (microns)')
ylabel('\rho_{tot}, \tau_{tot}, \alpha_{tot}')
legend('\rho_{tot}','\tau_{tot}','\alpha_{tot}')
legend('boxoff')
ylim([0 1])
hold on
end



