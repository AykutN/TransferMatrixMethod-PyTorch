function plot_TMM_Angle(layers, thickness, thick_film_index,angle,filename)

step = 10;
final_angle = 80;

for i = 0:step:final_angle
    
    angle = i;
    
    if angle == 0;
    angle_number = i;
    else
    angle_number = fix(angle/step);
    end
    
wavelength_min = 0.2 ; % microns
wavelength_max = 1.2; % microns
pts = 5e3+1; 


wL = linspace(wavelength_min,wavelength_max,pts)*1e-6; % wavelengths of light of interest
wL = wL';

t = thickness*1e-9; % in m
n = length(layers);

addpath(genpath([cd '/Materials/']));
N = zeros(length(wL),n);

for j = 1:n
    
    if strcmp(layers{j},'Vac') % assigns refractive index for Vacuum to 1
        N(:,j) = 1;
    elseif strcmp(layers{j},'InGaAs') % for InGaAs only
        x = 0.47;
        layer_props = str2func(layers{j});
        N(:,j) = layer_props(wL,x);    
    else
        layer_props = str2func(layers{j}); % Refers to material properties of 
        % chosen material for n and k values
        N(:,j) = layer_props(wL);
    end
    
end

R = zeros(length(wL),length(angle));
T = zeros(length(wL),length(angle));
A = zeros(length(wL),length(angle));

for ii = 1:length(angle)
% Thick Film Options. This part of the code is used when the thick film
% index is nonzero
if ~isempty(thick_film_index)
    
Composite_Layer_1 =  1:thick_film_index;
Composite_Layer_2 =  thick_film_index:n;

[R12s,R12p,T12s,T12p,angle_f] = TMM_Ref_Trans_Thick(wL, t(Composite_Layer_1), N(:,Composite_Layer_1),angle(ii));
[R23s,R23p,T23s,T23p] = TMM_Ref_Trans_Thick(wL, t(Composite_Layer_2), N(:,Composite_Layer_2),angle_f);

n_inc = N(:,1);
n_thick = N(:,thick_film_index);
k_thick = imag(N(:,thick_film_index));

alpha = (4*pi)./wL.*k_thick;
angle_film = (asin(n_inc./n_thick.*sin(angle(ii)*pi/180)))

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
% This is for thin-film optics
[R_tot,T_tot] = TMM_Ref_Trans(wL, t, N, angle(ii));
A_tot = 1 - R_tot - T_tot;

end
R(:,ii) = R_tot;%reflectivity
T(:,ii) = T_tot;%transmission  
A(:,ii) = A_tot;%absorption
end


%% AVT & Color Coordinates

path = [cd '/Materials/Properties/'];

layer_AM15G = load([path 'AM15G' '.txt']);
layer_RHE = load([path 'RHE' '.txt']);
layer_CMFs = load([path 'CMF_Lambda-x-y-z' '.txt']);
layer_D65 = load([path 'D65' '.txt']);

S_AM15G = interp1(layer_AM15G(:,1), layer_AM15G(:,2), wL*1E6, 'linear');
D65 = interp1(layer_D65(:,1), layer_D65(:,2), wL*1E6, 'linear');
V_RHE = interp1(layer_RHE(:,1), layer_RHE(:,2), wL*1E6, 'linear');
CMF_X = interp1(layer_CMFs(:,1), layer_CMFs(:,2), wL*1E6, 'linear');
CMF_Y = interp1(layer_CMFs(:,1), layer_CMFs(:,3), wL*1E6, 'linear');
CMF_Z = interp1(layer_CMFs(:,1), layer_CMFs(:,4), wL*1E6, 'linear');


PS = S_AM15G.*V_RHE;
PS(isnan(PS)) = 0;

Int_PS = trapz(wL*1E9,PS);

PST = PS.*T_tot;
PST(isnan(PST)) = 0;

Int_PST = trapz(wL*1E9,PST);

AVT = (Int_PST/Int_PS)*100;

maxT = max(T_tot);

CMF_X(isnan(CMF_X)) = 0;
CMF_Y(isnan(CMF_Y)) = 0;
CMF_Z(isnan(CMF_Z)) = 0;

STX = D65.*T_tot.*CMF_X;
STX(isnan(STX)) = 0;
XX = trapz(wL*1E9,STX);

STY = D65.*T_tot.*CMF_Y;
STY(isnan(STY)) = 0;
YY = trapz(wL*1E9,STY);

STZ = D65.*T_tot.*CMF_Z;
STZ(isnan(STZ)) = 0;
ZZ = trapz(wL*1E9,STZ);

x_cr = XX/(XX+YY+ZZ);
y_cr = YY/(XX+YY+ZZ);


header1 = {'Angle (Degree)','x', 'y', 'AVT (%)'};

num1 = [angle,x_cr,y_cr,AVT];

counter = angle_number+2;
angle_data(counter,:) = num2cell([num1]);

angle_data(1,:) = header1;
writecell(angle_data,filename,'sheet',3);


end

angle_data


end


