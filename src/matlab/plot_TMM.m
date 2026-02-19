function plot_TMM(layers, thickness, thick_film_index,angle,filename)

wavelength_min = 0.2; % microns
wavelength_max = 1.2; % microns
pts = 5e3+1; 


% wavelength_min = 0.525 ; % microns
% wavelength_max = 0.545 ; % microns
% pts = 5e3+1; 


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

PS = S_AM15G.*V_RHE; % for AVT
%PS = S_AM15G; % for AR
PS(isnan(PS)) = 0;

Int_PS = trapz(wL*1E9,PS);

PST = PS.*T_tot; % for AVT
%PST = PS.*R_tot; % for AR
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


%% Jph


 active_layer = load([path 'P3HT_PCBM' '.txt']);
 d_al = 100E-9; % nm
 %d_al = 0; % nm
 h = 4.13566766225e-15; % Planck's constant, h. [eVs]
 c = 2.9979e8; % Speed of light in vacuum. [m/s]

 k_al = interp1(active_layer(:,1), active_layer(:,3), wL*1E6, 'linear');
 k_al(isnan(k_al)) = 0;

 alpha_al = (4*pi)./wL.*k_al;

 
% Jph_integrant = T_tot.*S_AM15G.*(1-exp(-1.*alpha_al*d_al));
  Jph_integrant = A_tot.*S_AM15G;   % Jph (A_tot)


Jph_integrant(isnan(Jph_integrant)) = 0;
Jph = (1/(h*c))*trapz(wL*1E9,Jph_integrant)*100*1E-9; % mA/cm^2

%% Plot & Print

fprintf('Average Visible Transmittance = %4.4g',AVT)
%fprintf('Average Reflectance = %4.4g',AVT)
fprintf(' %% \n')
fprintf('Jph = %4.4g',Jph) % mA/cm^2
fprintf(' mA/cm^2 \n')
fprintf('Color Coordinates\n')
fprintf('x = %4.4g\n',x_cr)
fprintf('y = %4.4g\n',y_cr)

wL = wL*1e9;
figure()

plot(wL,R_tot,wL,T_tot,wL,A_tot,'Linewidth',3.0)
xlabel('Wavelength (nm)')
ylabel('R, T, A')
legend('R','T','A')
legend('boxoff')
ylim([0 1])
set(gca,'Fontsize',14,'FontWeight','bold','LineWidth',2.0)
grid on 


header1 = {'Wavelength (nm)','R', 'T', 'A'};
header2 = {'Average Visible Transmittance','Jph (mA/cm^2)','x','y',}; % for AVT
%header2 = {'Average Reflectance on AM1.5G','Jph (mA/cm^2)','x','y',}; % for AR
num = [wL,R_tot,T_tot,A_tot];
abc = [AVT,Jph,x_cr,y_cr,];
c = cell(5002,4);
c = {'Wavelength (nm)','R', 'T', 'A';wL',R_tot',T_tot',A_tot'};
d = cell(2,4);
d = {'Average Visible Transmittance', 'Jph (mA/cm^2)', 'x','y';AVT',Jph',x_cr',y_cr'}; %for AVT
%d = {'Average Reflectance on AM1.5G', 'Jph (mA/cm^2)', 'x', 'y';AVT',Jph',x_cr',y_cr'}; %for AR
d(1,:) = header2;
d(2,:) = num2cell([abc]);
c(1,:) = header1;
c(2:5002,:) = num2cell([num]);
e = cell(5005,4);
e = [d;c];
writecell(e,filename);

header3 = {'Layers','Thickness (nm)'};
column_number = size(layers,2);
structures = cell(column_number,1);
structures_t = cell(column_number,2);
structures = layers;
structures_t = thickness;
sheet2 = cell(column_number,2);
aa(:,1) = header3;
bb(1,:) = structures;
bb(2,:) = num2cell([structures_t]);
cc = horzcat(aa,bb);


writecell(cc,filename,'sheet',2);


end




