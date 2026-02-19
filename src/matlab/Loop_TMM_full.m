clear;clc;
angle = 0;
thick_film_index = [];

filename = "Loop_PTB7_maw.xlsx";

m=2; %independent variable

 for t1 = 1:1:50;
    for t2 = 1:1:20;
    %    for t3 = 1:1:50;
            % for t4 = [10];
            %     for t5 = [1 5 10 15 20 25 30 35 40 45 50];
            %         for t6 = [1 5 10 15 20 25 30 35 40 45 50];
        
     

layers(1) = {'Vac'}; thickness(1) = 1 ;%nm

layers(2) = {'ITO'}; thickness(2) =        50 ;%nm
layers(3) = {'ZnO'}; thickness(3) =        30 ;%nm
layers(4) = {'PTB7_PCBM'}; thickness(4) =  100 ;%nm
layers(5) = {'MoO3'}; thickness(5) =       10 ;%nm
layers(6) = {'Ag'}; thickness(6) =         t1 ;%nm
layers(7) = {'WO3'}; thickness(7) =        t2 ;%nm

layers(8) = {'Vac'}; thickness(8) =       1 ;%nm



% layers(1) = {'Vac'}; thickness(1) = 1 ;%nm
% 
% layers(2) = {'MoO3'}; thickness(2) =        t1                 ;%nm
% layers(3) = {'Ag'}; thickness(3) =          12                ;%nm
% layers(4) = {'MoO3'}; thickness(4) =        t2                 ;%nm
% layers(5) = {'P3HT_PCBM'}; thickness(5) =   50                 ;%nm
% layers(6) = {'ZnO'}; thickness(6) =         12                 ;%nm
% layers(7) = {'ITO'}; thickness(7) =         128                 ;%nm
% layers(8) = {'ZnO'}; thickness(8) =         14                 ;%nm
% layers(9) = {'PBDB_T_ITIC'}; thickness(9) = 50                 ;%nm
% layers(10) = {'MoO3'}; thickness(10) =      20                 ;%nm
% layers(11) = {'Ag'}; thickness(11) =         5                 ;%nm
% layers(12) = {'MoO3'}; thickness(12) =      26                 ;%nm
% 
% layers(13) = {'Vac'}; thickness(13) =     1 ;%nm

% clf;plot_TMM(layers, thickness, thick_film_index,angle,filename);
% clf;plot_TMM_Angle(layers, thickness, thick_film_index,angle,filename);
% clf;plot_TMM_CRI(layers, thickness, thick_film_index,angle,filename);


%%
wavelength_min = 0.2 ; % microns
wavelength_max = 1.2; % microns
pts = 5e3+1; 


% wavelength_min = 0.53 ; % microns
% wavelength_max = 0.63 ; % microns
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
layer_TCSs = load([path 'Test_Color_Samples' '.txt']);
layer_Ss = load([path 'SforD', '.txt']);

S_AM15G = interp1(layer_AM15G(:,1), layer_AM15G(:,2), wL*1E6, 'linear');
D65 = interp1(layer_D65(:,1), layer_D65(:,2), wL*1E6, 'linear');
V_RHE = interp1(layer_RHE(:,1), layer_RHE(:,2), wL*1E6, 'linear');
CMF_X = interp1(layer_CMFs(:,1), layer_CMFs(:,2), wL*1E6, 'linear');
CMF_Y = interp1(layer_CMFs(:,1), layer_CMFs(:,3), wL*1E6, 'linear');
CMF_Z = interp1(layer_CMFs(:,1), layer_CMFs(:,4), wL*1E6, 'linear');

TCS1 = interp1(layer_TCSs(:,1), layer_TCSs(:,2), wL*1E6, 'linear');
TCS2 = interp1(layer_TCSs(:,1), layer_TCSs(:,3), wL*1E6, 'linear');
TCS3 = interp1(layer_TCSs(:,1), layer_TCSs(:,4), wL*1E6, 'linear');
TCS4 = interp1(layer_TCSs(:,1), layer_TCSs(:,5), wL*1E6, 'linear');
TCS5 = interp1(layer_TCSs(:,1), layer_TCSs(:,6), wL*1E6, 'linear');
TCS6 = interp1(layer_TCSs(:,1), layer_TCSs(:,7), wL*1E6, 'linear');
TCS7 = interp1(layer_TCSs(:,1), layer_TCSs(:,8), wL*1E6, 'linear');
TCS8 = interp1(layer_TCSs(:,1), layer_TCSs(:,9), wL*1E6, 'linear');
TCS9 = interp1(layer_TCSs(:,1), layer_TCSs(:,10), wL*1E6, 'linear');
TCS10 = interp1(layer_TCSs(:,1), layer_TCSs(:,11), wL*1E6, 'linear');
TCS11 = interp1(layer_TCSs(:,1), layer_TCSs(:,12), wL*1E6, 'linear');
TCS12 = interp1(layer_TCSs(:,1), layer_TCSs(:,13), wL*1E6, 'linear');
TCS13 = interp1(layer_TCSs(:,1), layer_TCSs(:,14), wL*1E6, 'linear');
TCS14 = interp1(layer_TCSs(:,1), layer_TCSs(:,15), wL*1E6, 'linear');
TCS15 = interp1(layer_TCSs(:,1), layer_TCSs(:,16), wL*1E6, 'linear');

S0=layer_Ss(:,2);
S1=layer_Ss(:,3);
S2=layer_Ss(:,4);

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

%% From salt CIE 1931 chromacity coordinates to salt CIE 1960 coordinates:
TX = T_tot.*CMF_X;
TX(isnan(TX)) = 0;
XX_ = trapz(wL*1E9,TX);

TY = T_tot.*CMF_Y;
TY(isnan(TY)) = 0;
YY_ = trapz(wL*1E9,TY);

TZ = T_tot.*CMF_Z;
TZ(isnan(TZ)) = 0;
ZZ_ = trapz(wL*1E9,TZ);

x_salt = XX_/(XX_+YY_+ZZ_);
y_salt = YY_/(XX_+YY_+ZZ_);

U_=(2/3)*XX_;
V_=YY_;
W_=(1/2)*(-XX_+3*YY_+ZZ_);
u_salt=U_/(U_+V_+W_);
v_salt=V_/(U_+V_+W_);

c_k=(1/v_salt)*(4-u_salt-10*v_salt);
d_k=(1/v_salt)*(1.708*v_salt+0.404-1.481*u_salt);
	 
%% CCT by using McCamy's approximation algorithm to estimate the CCT from the xy chromaticities:
xe = 0.3320;
ye = 0.1858;
n=(x_salt-xe)/(y_salt-ye); %inverse slope line, and (xe = 0.3320, ye = 0.1858) is the "epicenter"; quite close to the intersection point mentioned by Kelly. The maximum absolute error for color temperatures ranging from 2856 K (illuminant A) to 6504 K (D65) is under 2 K.
CCT=-(449*n^3)+(3525*n^2)-(6823.3*n^1)+(5520.33*n^0);

%% Reference calculation from Black Body Radiation:
c_=299792458;
h_=6.6260694E-34;
k_=1.3806565E-23;
Tc=CCT;

%	for i=1:5001
%	DB(i)=((i*0.2)+199.8)*1E-9;
%	ref(i)=(2*pi*h_*(c_^2)*((DB(i))^(-5)))/((exp((h_*c_)/(k_*Tc*DB(i))))-1);
%   end
%  ref_e=(ref/max(ref));
%  S_ref=ref_e';
 
if Tc<=5100;
    
    for i=1:5001
    
    DB(i)=((i*0.2)+199.8)*1E-9;
    ref(i)=(2*pi*h_*(c_^2)*((DB(i))^(-5)))/((exp((h_*c_)/(k_*Tc*DB(i))))-1);
     
    end
    ref_e=(ref/max(ref));
    S_ref=ref_e';    
    
elseif Tc<=7000;
    kd=(-4.6070E9/(Tc^3))+(2.9678E6/(Tc^2))+(0.09911E3/(Tc))+0.244063;
    ld=-3.000*(kd^2)+2.870*kd-0.275;
    M1=(-1.3515-1.7703*kd+5.9114*ld)/(0.0241+0.2562*kd-0.7341*ld);
    M2=(0.0300-31.4424*kd+30.0717*ld)/(0.0241+0.2562*kd-0.7341*ld);
    S_ref=S0+(M1*S1)+(M2*S2);
    S_ref=S_ref/max(S_ref);
    
else
    kd=(-2.00640E9/(Tc^3))+(1.9018E6/(Tc^2))+(0.24748E3/(Tc))+0.237040;
    ld=-3.000*(kd^2)+2.870*kd-0.275;
    M1=(-1.3515-1.7703*kd+5.9114*ld)/(0.0241+0.2562*kd-0.7341*ld);
    M2=(0.0300-31.4424*kd+30.0717*ld)/(0.0241+0.2562*kd-0.7341*ld);
    S_ref=S0+(M1*S1)+(M2*S2);
    S_ref=S_ref/max(S_ref);
        
end
 

%% Calculation reference chromacity coordinates:
S_refX = S_ref.*CMF_X;
S_refX(isnan(S_refX)) = 0;
XX_ref = trapz(wL*1E9,S_refX);

S_refY = S_ref.*CMF_Y;
S_refY (isnan(S_refY)) = 0;
YY_ref = trapz(wL*1E9,S_refY);

S_refZ = S_ref.*CMF_Z;
S_refZ(isnan(S_refZ)) = 0;
ZZ_ref = trapz(wL*1E9, S_refZ);

x_ref = XX_ref/(XX_ref+YY_ref+ZZ_ref);
y_ref = YY_ref/(XX_ref+YY_ref+ZZ_ref);

U_ref=(2/3)*XX_ref;
V_ref=YY_ref;
W_ref=(1/2)*(-XX_ref+3*YY_ref+ZZ_ref);
u_ref=U_ref/(U_ref+V_ref+W_ref);
v_ref=V_ref/(U_ref+V_ref+W_ref);

c_r=(1/v_ref)*(4-u_ref-10*v_ref);
d_r=(1/v_ref)*(1.708*v_ref+0.404-1.481*u_ref);


%% Coordinate of Planckian locus (0.2528, 0.3484) for CIE1960:

DC=sqrt(((u_salt-u_ref)^2)+((v_salt-v_ref)^2));   %The distance of the test point to the reference locus (must be <(5.4E-3))

%% Calculation of adaptive shift due to the different state of chromatic adaption under test lamp (r) and reference illimunation
uk_=u_ref;
vk_=v_ref;

%% From salt CIE 1931 chromacity coordinates to salt CIE 1960 coordinates for reference and TCS
%% 1
S_refX1 = S_ref.*CMF_X.*TCS1;
S_refX1(isnan(S_refX1)) = 0;
XX_ref1 = trapz(wL*1E9,S_refX1);

S_refY1 = S_ref.*CMF_Y.*TCS1;
S_refY1 (isnan(S_refY1)) = 0;
YY_ref1 = trapz(wL*1E9,S_refY1);

S_refZ1 = S_ref.*CMF_Z.*TCS1;
S_refZ1(isnan(S_refZ1)) = 0;
ZZ_ref1 = trapz(wL*1E9, S_refZ1);

% x_ref1 = XX_ref1/(XX_ref1+YY_ref1+ZZ_ref1);
% y_ref1 = YY_ref1/(XX_ref1+YY_ref1+ZZ_ref1);
% V_ref1=YY_ref1;
U_ref1=(2/3)*XX_ref1;
W_ref1=(1/2)*(-XX_ref1+3*YY_ref1+ZZ_ref1);
u_ref1=U_ref1/(U_ref1+YY_ref1+W_ref1);
v_ref1=YY_ref1/(U_ref1+YY_ref1+W_ref1);

% From salt CIE 1931 chromacity coordinates to salt CIE 1960 coordinates test and TCS
TX1=T_tot.*CMF_X.*TCS1;
TX1(isnan(TX1)) = 0;
XX_1 = trapz(wL*1E9,TX1);

TY1 = T_tot.*CMF_Y.*TCS1;
TY1(isnan(TY1)) = 0;
YY_1 = trapz(wL*1E9,TY1);

TZ1 = T_tot.*CMF_Z.*TCS1;
TZ1(isnan(TZ1)) = 0;
ZZ_1 = trapz(wL*1E9,TZ1);
 
% x_t1 = XX_1/(XX_1+YY_1+ZZ_1);
% y_t1 = YY_1/(XX_1+YY_1+ZZ_1);
% V_1=YY_1;

U_1=(2/3)*XX_1;
W_1=(1/2)*(-XX_1+3*YY_1+ZZ_1);
u_t1=U_1/(U_1+YY_1+W_1);
v_t1=YY_1/(U_1+YY_1+W_1);

c_k1=(1/v_t1)*(4-u_t1-10*v_t1);
d_k1=(1/v_t1)*(1.708*v_t1+0.404-1.481*u_t1);

% Calculation of adaptive shift due to the different state of chromatic adaption under test lamp (r) and reference (r) illimunation
uk1_=(10.872+0.404*(c_r/c_k)*c_k1-4*(d_r/d_k)*d_k1)/(16.518+1.481*(c_r/c_k)*c_k1-(d_r/d_k)*d_k1);
vk1_=(5.520)/(16.518+1.481*(c_r/c_k)*c_k1-(d_r/d_k)*d_k1);
Wr1=25*(((100/YY_ref)*YY_ref1)^(1/3))-17;
Ur1=13*Wr1*(u_ref1-u_ref);
Vr1=13*Wr1*(v_ref1-v_ref);
Wk1=25*(((100/YY_)*YY_1)^(1/3))-17;
Uk1=13*Wk1*(uk1_-uk_);
Vk1=13*Wk1*(vk1_-vk_);

%% 2
S_refX2 = S_ref.*CMF_X.*TCS2;
S_refX2(isnan(S_refX2)) = 0;
XX_ref2 = trapz(wL*1E9,S_refX2);

S_refY2 = S_ref.*CMF_Y.*TCS2;
S_refY2 (isnan(S_refY2)) = 0;
YY_ref2 = trapz(wL*1E9,S_refY2);

S_refZ2 = S_ref.*CMF_Z.*TCS2;
S_refZ2(isnan(S_refZ2)) = 0;
ZZ_ref2 = trapz(wL*1E9, S_refZ2);

U_ref2=(2/3)*XX_ref2;
W_ref2=(1/2)*(-XX_ref2+3*YY_ref2+ZZ_ref2);
u_ref2=U_ref2/(U_ref2+YY_ref2+W_ref2);
v_ref2=YY_ref2/(U_ref2+YY_ref2+W_ref2);

% From salt CIE 1931 chromacity coordinates to salt CIE 1960 coordinates test and TCS
TX2=T_tot.*CMF_X.*TCS2;
TX2(isnan(TX2)) = 0;
XX_2 = trapz(wL*1E9,TX2);

TY2 = T_tot.*CMF_Y.*TCS2;
TY2(isnan(TY2)) = 0;
YY_2 = trapz(wL*1E9,TY2);

TZ2 = T_tot.*CMF_Z.*TCS2;
TZ2(isnan(TZ2)) = 0;
ZZ_2 = trapz(wL*1E9,TZ2);
 
U_2=(2/3)*XX_2;
W_2=(1/2)*(-XX_2+3*YY_2+ZZ_2);
u_t2=U_2/(U_2+YY_2+W_2);
v_t2=YY_2/(U_2+YY_2+W_2);

c_k2=(1/v_t2)*(4-u_t2-10*v_t2);
d_k2=(1/v_t2)*(1.708*v_t2+0.404-1.481*u_t2);

uk2_=(10.872+0.404*(c_r/c_k)*c_k2-4*(d_r/d_k)*d_k2)/(16.518+1.481*(c_r/c_k)*c_k2-(d_r/d_k)*d_k2);
vk2_=(5.520)/(16.518+1.481*(c_r/c_k)*c_k2-(d_r/d_k)*d_k2);
Wr2=25*(((100/YY_ref)*YY_ref2)^(1/3))-17;
Ur2=13*Wr2*(u_ref2-u_ref);
Vr2=13*Wr2*(v_ref2-v_ref);
Wk2=25*(((100/YY_)*YY_2)^(1/3))-17;
Uk2=13*Wk2*(uk2_-uk_);
Vk2=13*Wk2*(vk2_-vk_);

%% 3
S_refX3 = S_ref.*CMF_X.*TCS3;
S_refX3(isnan(S_refX3)) = 0;
XX_ref3 = trapz(wL*1E9,S_refX3);
 
S_refY3 = S_ref.*CMF_Y.*TCS3;
S_refY3 (isnan(S_refY3)) = 0;
YY_ref3 = trapz(wL*1E9,S_refY3);
 
S_refZ3 = S_ref.*CMF_Z.*TCS3;
S_refZ3(isnan(S_refZ3)) = 0;
ZZ_ref3 = trapz(wL*1E9, S_refZ3);
 
U_ref3=(2/3)*XX_ref3;
W_ref3=(1/2)*(-XX_ref3+3*YY_ref3+ZZ_ref3);
u_ref3=U_ref3/(U_ref3+YY_ref3+W_ref3);
v_ref3=YY_ref3/(U_ref3+YY_ref3+W_ref3);

% From salt CIE 1931 chromacity coordinates to salt CIE 1960 coordinates test and TCS
TX3=T_tot.*CMF_X.*TCS3;
TX3(isnan(TX3)) = 0;
XX_3 = trapz(wL*1E9,TX3);
 
TY3 = T_tot.*CMF_Y.*TCS3;
TY3(isnan(TY3)) = 0;
YY_3 = trapz(wL*1E9,TY3);
 
TZ3 = T_tot.*CMF_Z.*TCS3;
TZ3(isnan(TZ3)) = 0;
ZZ_3 = trapz(wL*1E9,TZ3);
 
U_3=(2/3)*XX_3;
W_3=(1/2)*(-XX_3+3*YY_3+ZZ_3);
u_t3=U_3/(U_3+YY_3+W_3);
v_t3=YY_3/(U_3+YY_3+W_3);
 
c_k3=(1/v_t3)*(4-u_t3-10*v_t3);
d_k3=(1/v_t3)*(1.708*v_t3+0.404-1.481*u_t3);

uk3_=(10.872+0.404*(c_r/c_k)*c_k3-4*(d_r/d_k)*d_k3)/(16.518+1.481*(c_r/c_k)*c_k3-(d_r/d_k)*d_k3);
vk3_=(5.520)/(16.518+1.481*(c_r/c_k)*c_k3-(d_r/d_k)*d_k3);
Wr3=25*(((100/YY_ref)*YY_ref3)^(1/3))-17;
Ur3=13*Wr3*(u_ref3-u_ref);
Vr3=13*Wr3*(v_ref3-v_ref);
Wk3=25*(((100/YY_)*YY_3)^(1/3))-17;
Uk3=13*Wk3*(uk3_-uk_);
Vk3=13*Wk3*(vk3_-vk_);


%% 4
S_refX4 = S_ref.*CMF_X.*TCS4;
S_refX4(isnan(S_refX4)) = 0;
XX_ref4 = trapz(wL*1E9,S_refX4);
 
S_refY4 = S_ref.*CMF_Y.*TCS4;
S_refY4 (isnan(S_refY4)) = 0;
YY_ref4 = trapz(wL*1E9,S_refY4);
 
S_refZ4 = S_ref.*CMF_Z.*TCS4;
S_refZ4(isnan(S_refZ4)) = 0;
ZZ_ref4 = trapz(wL*1E9, S_refZ4);
 
U_ref4=(2/3)*XX_ref4;
W_ref4=(1/2)*(-XX_ref4+3*YY_ref4+ZZ_ref4);
u_ref4=U_ref4/(U_ref4+YY_ref4+W_ref4);
v_ref4=YY_ref4/(U_ref4+YY_ref4+W_ref4);

TX4=T_tot.*CMF_X.*TCS4;
TX4(isnan(TX4)) = 0;
XX_4 = trapz(wL*1E9,TX4);
 
TY4 = T_tot.*CMF_Y.*TCS4;
TY4(isnan(TY4)) = 0;
YY_4 = trapz(wL*1E9,TY4);
 
TZ4 = T_tot.*CMF_Z.*TCS4;
TZ4(isnan(TZ4)) = 0;
ZZ_4 = trapz(wL*1E9,TZ4);
 
U_4=(2/3)*XX_4;
W_4=(1/2)*(-XX_4+3*YY_4+ZZ_4);
u_t4=U_4/(U_4+YY_4+W_4);
v_t4=YY_4/(U_4+YY_4+W_4);
 
c_k4=(1/v_t4)*(4-u_t4-10*v_t4);
d_k4=(1/v_t4)*(1.708*v_t4+0.404-1.481*u_t4);

uk4_=(10.872+0.404*(c_r/c_k)*c_k4-4*(d_r/d_k)*d_k4)/(16.518+1.481*(c_r/c_k)*c_k4-(d_r/d_k)*d_k4);
vk4_=(5.520)/(16.518+1.481*(c_r/c_k)*c_k4-(d_r/d_k)*d_k4);
Wr4=25*(((100/YY_ref)*YY_ref4)^(1/3))-17;
Ur4=13*Wr4*(u_ref4-u_ref);
Vr4=13*Wr4*(v_ref4-v_ref);
Wk4=25*(((100/YY_)*YY_4)^(1/3))-17;
Uk4=13*Wk4*(uk4_-uk_);
Vk4=13*Wk4*(vk4_-vk_);

%% 5
S_refX5 = S_ref.*CMF_X.*TCS5;
S_refX5(isnan(S_refX5)) = 0;
XX_ref5 = trapz(wL*1E9,S_refX5);
 
S_refY5 = S_ref.*CMF_Y.*TCS5;
S_refY5 (isnan(S_refY5)) = 0;
YY_ref5 = trapz(wL*1E9,S_refY5);
 
S_refZ5 = S_ref.*CMF_Z.*TCS5;
S_refZ5(isnan(S_refZ5)) = 0;
ZZ_ref5 = trapz(wL*1E9, S_refZ5);
 
U_ref5=(2/3)*XX_ref5;
W_ref5=(1/2)*(-XX_ref5+3*YY_ref5+ZZ_ref5);
u_ref5=U_ref5/(U_ref5+YY_ref5+W_ref5);
v_ref5=YY_ref5/(U_ref5+YY_ref5+W_ref5);


TX5=T_tot.*CMF_X.*TCS5;
TX5(isnan(TX5)) = 0;
XX_5 = trapz(wL*1E9,TX5);
 
TY5 = T_tot.*CMF_Y.*TCS5;
TY5(isnan(TY5)) = 0;
YY_5 = trapz(wL*1E9,TY5);
 
TZ5 = T_tot.*CMF_Z.*TCS5;
TZ5(isnan(TZ5)) = 0;
ZZ_5 = trapz(wL*1E9,TZ5);
 
U_5=(2/3)*XX_5;
W_5=(1/2)*(-XX_5+3*YY_5+ZZ_5);
u_t5=U_5/(U_5+YY_5+W_5);
v_t5=YY_5/(U_5+YY_5+W_5);
 
c_k5=(1/v_t5)*(4-u_t5-10*v_t5);
d_k5=(1/v_t5)*(1.708*v_t5+0.404-1.481*u_t5);

uk5_=(10.872+0.404*(c_r/c_k)*c_k5-4*(d_r/d_k)*d_k5)/(16.518+1.481*(c_r/c_k)*c_k5-(d_r/d_k)*d_k5);
vk5_=(5.520)/(16.518+1.481*(c_r/c_k)*c_k5-(d_r/d_k)*d_k5);
Wr5=25*(((100/YY_ref)*YY_ref5)^(1/3))-17;
Ur5=13*Wr5*(u_ref5-u_ref);
Vr5=13*Wr5*(v_ref5-v_ref);
Wk5=25*(((100/YY_)*YY_5)^(1/3))-17;
Uk5=13*Wk5*(uk5_-uk_);
Vk5=13*Wk5*(vk5_-vk_);

%% 6
S_refX6 = S_ref.*CMF_X.*TCS6;
S_refX6(isnan(S_refX6)) = 0;
XX_ref6 = trapz(wL*1E9,S_refX6);
 
S_refY6 = S_ref.*CMF_Y.*TCS6;
S_refY6 (isnan(S_refY6)) = 0;
YY_ref6 = trapz(wL*1E9,S_refY6);
 
S_refZ6 = S_ref.*CMF_Z.*TCS6;
S_refZ6(isnan(S_refZ6)) = 0;
ZZ_ref6 = trapz(wL*1E9, S_refZ6);
 
U_ref6=(2/3)*XX_ref6;
W_ref6=(1/2)*(-XX_ref6+3*YY_ref6+ZZ_ref6);
u_ref6=U_ref6/(U_ref6+YY_ref6+W_ref6);
v_ref6=YY_ref6/(U_ref6+YY_ref6+W_ref6);

TX6=T_tot.*CMF_X.*TCS6;
TX6(isnan(TX6)) = 0;
XX_6 = trapz(wL*1E9,TX6);
 
TY6 = T_tot.*CMF_Y.*TCS6;
TY6(isnan(TY6)) = 0;
YY_6 = trapz(wL*1E9,TY6);
 
TZ6 = T_tot.*CMF_Z.*TCS6;
TZ6(isnan(TZ6)) = 0;
ZZ_6 = trapz(wL*1E9,TZ6);
 
U_6=(2/3)*XX_6;
W_6=(1/2)*(-XX_6+3*YY_6+ZZ_6);
u_t6=U_6/(U_6+YY_6+W_6);
v_t6=YY_6/(U_6+YY_6+W_6);
 
c_k6=(1/v_t6)*(4-u_t6-10*v_t6);
d_k6=(1/v_t6)*(1.708*v_t6+0.404-1.481*u_t6);


uk6_=(10.872+0.404*(c_r/c_k)*c_k6-4*(d_r/d_k)*d_k6)/(16.518+1.481*(c_r/c_k)*c_k6-(d_r/d_k)*d_k6);
vk6_=(5.520)/(16.518+1.481*(c_r/c_k)*c_k6-(d_r/d_k)*d_k6);
Wr6=25*(((100/YY_ref)*YY_ref6)^(1/3))-17;
Ur6=13*Wr6*(u_ref6-u_ref);
Vr6=13*Wr6*(v_ref6-v_ref);
Wk6=25*(((100/YY_)*YY_6)^(1/3))-17;
Uk6=13*Wk6*(uk6_-uk_);
Vk6=13*Wk6*(vk6_-vk_);


%% 7
S_refX7 = S_ref.*CMF_X.*TCS7;
S_refX7(isnan(S_refX7)) = 0;
XX_ref7 = trapz(wL*1E9,S_refX7);
 
S_refY7 = S_ref.*CMF_Y.*TCS7;
S_refY7 (isnan(S_refY7)) = 0;
YY_ref7 = trapz(wL*1E9,S_refY7);
 
S_refZ7 = S_ref.*CMF_Z.*TCS7;
S_refZ7(isnan(S_refZ7)) = 0;
ZZ_ref7 = trapz(wL*1E9, S_refZ7);
 
U_ref7=(2/3)*XX_ref7;
W_ref7=(1/2)*(-XX_ref7+3*YY_ref7+ZZ_ref7);
u_ref7=U_ref7/(U_ref7+YY_ref7+W_ref7);
v_ref7=YY_ref7/(U_ref7+YY_ref7+W_ref7);

TX7=T_tot.*CMF_X.*TCS7;
TX7(isnan(TX7)) = 0;
XX_7 = trapz(wL*1E9,TX7);
 
TY7 = T_tot.*CMF_Y.*TCS7;
TY7(isnan(TY7)) = 0;
YY_7 = trapz(wL*1E9,TY7);
 
TZ7 = T_tot.*CMF_Z.*TCS7;
TZ7(isnan(TZ7)) = 0;
ZZ_7 = trapz(wL*1E9,TZ7);
 
U_7=(2/3)*XX_7;
W_7=(1/2)*(-XX_7+3*YY_7+ZZ_7);
u_t7=U_7/(U_7+YY_7+W_7);
v_t7=YY_7/(U_7+YY_7+W_7);
 
c_k7=(1/v_t7)*(4-u_t7-10*v_t7);
d_k7=(1/v_t7)*(1.708*v_t7+0.404-1.481*u_t7);


uk7_=(10.872+0.404*(c_r/c_k)*c_k7-4*(d_r/d_k)*d_k7)/(16.518+1.481*(c_r/c_k)*c_k7-(d_r/d_k)*d_k7);
vk7_=(5.520)/(16.518+1.481*(c_r/c_k)*c_k7-(d_r/d_k)*d_k7);
Wr7=25*(((100/YY_ref)*YY_ref7)^(1/3))-17;
Ur7=13*Wr7*(u_ref7-u_ref);
Vr7=13*Wr7*(v_ref7-v_ref);
Wk7=25*(((100/YY_)*YY_7)^(1/3))-17;
Uk7=13*Wk7*(uk7_-uk_);
Vk7=13*Wk7*(vk7_-vk_);


%% 8
S_refX8 = S_ref.*CMF_X.*TCS8;
S_refX8(isnan(S_refX8)) = 0;
XX_ref8 = trapz(wL*1E9,S_refX8);
 
S_refY8 = S_ref.*CMF_Y.*TCS8;
S_refY8 (isnan(S_refY8)) = 0;
YY_ref8 = trapz(wL*1E9,S_refY8);
 
S_refZ8 = S_ref.*CMF_Z.*TCS8;
S_refZ8(isnan(S_refZ8)) = 0;
ZZ_ref8 = trapz(wL*1E9, S_refZ8);
 
U_ref8=(2/3)*XX_ref8;
W_ref8=(1/2)*(-XX_ref8+3*YY_ref8+ZZ_ref8);
u_ref8=U_ref8/(U_ref8+YY_ref8+W_ref8);
v_ref8=YY_ref8/(U_ref8+YY_ref8+W_ref8);

TX8=T_tot.*CMF_X.*TCS8;
TX8(isnan(TX8)) = 0;
XX_8 = trapz(wL*1E9,TX8);
 
TY8 = T_tot.*CMF_Y.*TCS8;
TY8(isnan(TY8)) = 0;
YY_8 = trapz(wL*1E9,TY8);
 
TZ8 = T_tot.*CMF_Z.*TCS8;
TZ8(isnan(TZ8)) = 0;
ZZ_8 = trapz(wL*1E9,TZ8);
 
U_8=(2/3)*XX_8;
W_8=(1/2)*(-XX_8+3*YY_8+ZZ_8);
u_t8=U_8/(U_8+YY_8+W_8);
v_t8=YY_8/(U_8+YY_8+W_8);
 
c_k8=(1/v_t8)*(4-u_t8-10*v_t8);
d_k8=(1/v_t8)*(1.708*v_t8+0.404-1.481*u_t8);


uk8_=(10.872+0.404*(c_r/c_k)*c_k8-4*(d_r/d_k)*d_k8)/(16.518+1.481*(c_r/c_k)*c_k8-(d_r/d_k)*d_k8);
vk8_=(5.520)/(16.518+1.481*(c_r/c_k)*c_k8-(d_r/d_k)*d_k8);
Wr8=25*(((100/YY_ref)*YY_ref8)^(1/3))-17;
Ur8=13*Wr8*(u_ref8-u_ref);
Vr8=13*Wr8*(v_ref8-v_ref);
Wk8=25*(((100/YY_)*YY_8)^(1/3))-17;
Uk8=13*Wk8*(uk8_-uk_);
Vk8=13*Wk8*(vk8_-vk_);


%% 9
S_refX9 = S_ref.*CMF_X.*TCS9;
S_refX9(isnan(S_refX9)) = 0;
XX_ref9 = trapz(wL*1E9,S_refX9);
 
S_refY9 = S_ref.*CMF_Y.*TCS9;
S_refY9 (isnan(S_refY9)) = 0;
YY_ref9 = trapz(wL*1E9,S_refY9);
 
S_refZ9 = S_ref.*CMF_Z.*TCS9;
S_refZ9(isnan(S_refZ9)) = 0;
ZZ_ref9 = trapz(wL*1E9, S_refZ9);
 
U_ref9=(2/3)*XX_ref9;
W_ref9=(1/2)*(-XX_ref9+3*YY_ref9+ZZ_ref9);
u_ref9=U_ref9/(U_ref9+YY_ref9+W_ref9);
v_ref9=YY_ref9/(U_ref9+YY_ref9+W_ref9);

TX9=T_tot.*CMF_X.*TCS9;
TX9(isnan(TX9)) = 0;
XX_9 = trapz(wL*1E9,TX9);
 
TY9 = T_tot.*CMF_Y.*TCS9;
TY9(isnan(TY9)) = 0;
YY_9 = trapz(wL*1E9,TY9);
 
TZ9 = T_tot.*CMF_Z.*TCS9;
TZ9(isnan(TZ9)) = 0;
ZZ_9 = trapz(wL*1E9,TZ9);
 
U_9=(2/3)*XX_9;
W_9=(1/2)*(-XX_9+3*YY_9+ZZ_9);
u_t9=U_9/(U_9+YY_9+W_9);
v_t9=YY_9/(U_9+YY_9+W_9);
 
c_k9=(1/v_t9)*(4-u_t9-10*v_t9);
d_k9=(1/v_t9)*(1.708*v_t9+0.404-1.481*u_t9);


uk9_=(10.872+0.404*(c_r/c_k)*c_k9-4*(d_r/d_k)*d_k9)/(16.518+1.481*(c_r/c_k)*c_k9-(d_r/d_k)*d_k9);
vk9_=(5.520)/(16.518+1.481*(c_r/c_k)*c_k9-(d_r/d_k)*d_k9);
Wr9=25*(((100/YY_ref)*YY_ref9)^(1/3))-17;
Ur9=13*Wr9*(u_ref9-u_ref);
Vr9=13*Wr9*(v_ref9-v_ref);
Wk9=25*(((100/YY_)*YY_9)^(1/3))-17;
Uk9=13*Wk9*(uk9_-uk_);
Vk9=13*Wk9*(vk9_-vk_);

%% 10
S_refX10 = S_ref.*CMF_X.*TCS10;
S_refX10(isnan(S_refX10)) = 0;
XX_ref10 = trapz(wL*1E9,S_refX10);
 
S_refY10 = S_ref.*CMF_Y.*TCS10;
S_refY10 (isnan(S_refY10)) = 0;
YY_ref10 = trapz(wL*1E9,S_refY10);
 
S_refZ10 = S_ref.*CMF_Z.*TCS10;
S_refZ10(isnan(S_refZ10)) = 0;
ZZ_ref10 = trapz(wL*1E9, S_refZ10);
 
U_ref10=(2/3)*XX_ref10;
W_ref10=(1/2)*(-XX_ref10+3*YY_ref10+ZZ_ref10);
u_ref10=U_ref10/(U_ref10+YY_ref10+W_ref10);
v_ref10=YY_ref10/(U_ref10+YY_ref10+W_ref10);

TX10=T_tot.*CMF_X.*TCS10;
TX10(isnan(TX10)) = 0;
XX_10 = trapz(wL*1E9,TX10);
 
TY10 = T_tot.*CMF_Y.*TCS10;
TY10(isnan(TY10)) = 0;
YY_10 = trapz(wL*1E9,TY10);
 
TZ10 = T_tot.*CMF_Z.*TCS10;
TZ10(isnan(TZ10)) = 0;
ZZ_10 = trapz(wL*1E9,TZ10);
 
U_10=(2/3)*XX_10;
W_10=(1/2)*(-XX_10+3*YY_10+ZZ_10);
u_t10=U_10/(U_10+YY_10+W_10);
v_t10=YY_10/(U_10+YY_10+W_10);
 
c_k10=(1/v_t10)*(4-u_t10-10*v_t10);
d_k10=(1/v_t10)*(1.708*v_t10+0.404-1.481*u_t10);


uk10_=(10.872+0.404*(c_r/c_k)*c_k10-4*(d_r/d_k)*d_k10)/(16.518+1.481*(c_r/c_k)*c_k10-(d_r/d_k)*d_k10);
vk10_=(5.520)/(16.518+1.481*(c_r/c_k)*c_k10-(d_r/d_k)*d_k10);
Wr10=25*(((100/YY_ref)*YY_ref10)^(1/3))-17;
Ur10=13*Wr10*(u_ref10-u_ref);
Vr10=13*Wr10*(v_ref10-v_ref);
Wk10=25*(((100/YY_)*YY_10)^(1/3))-17;
Uk10=13*Wk10*(uk10_-uk_);
Vk10=13*Wk10*(vk10_-vk_);


%% 11
S_refX11 = S_ref.*CMF_X.*TCS11;
S_refX11(isnan(S_refX11)) = 0;
XX_ref11 = trapz(wL*1E9,S_refX11);
 
S_refY11 = S_ref.*CMF_Y.*TCS11;
S_refY11 (isnan(S_refY11)) = 0;
YY_ref11 = trapz(wL*1E9,S_refY11);
 
S_refZ11 = S_ref.*CMF_Z.*TCS11;
S_refZ11(isnan(S_refZ11)) = 0;
ZZ_ref11 = trapz(wL*1E9, S_refZ11);
 
U_ref11=(2/3)*XX_ref11;
W_ref11=(1/2)*(-XX_ref11+3*YY_ref11+ZZ_ref11);
u_ref11=U_ref11/(U_ref11+YY_ref11+W_ref11);
v_ref11=YY_ref11/(U_ref11+YY_ref11+W_ref11);

TX11=T_tot.*CMF_X.*TCS11;
TX11(isnan(TX11)) = 0;
XX_11 = trapz(wL*1E9,TX11);
 
TY11 = T_tot.*CMF_Y.*TCS11;
TY11(isnan(TY11)) = 0;
YY_11 = trapz(wL*1E9,TY11);
 
TZ11 = T_tot.*CMF_Z.*TCS11;
TZ11(isnan(TZ11)) = 0;
ZZ_11 = trapz(wL*1E9,TZ11);
 
U_11=(2/3)*XX_11;
W_11=(1/2)*(-XX_11+3*YY_11+ZZ_11);
u_t11=U_11/(U_11+YY_11+W_11);
v_t11=YY_11/(U_11+YY_11+W_11);
 
c_k11=(1/v_t11)*(4-u_t11-10*v_t11);
d_k11=(1/v_t11)*(1.708*v_t11+0.404-1.481*u_t11);


uk11_=(10.872+0.404*(c_r/c_k)*c_k11-4*(d_r/d_k)*d_k11)/(16.518+1.481*(c_r/c_k)*c_k11-(d_r/d_k)*d_k11);
vk11_=(5.520)/(16.518+1.481*(c_r/c_k)*c_k11-(d_r/d_k)*d_k11);
Wr11=25*(((100/YY_ref)*YY_ref11)^(1/3))-17;
Ur11=13*Wr11*(u_ref11-u_ref);
Vr11=13*Wr11*(v_ref11-v_ref);
Wk11=25*(((100/YY_)*YY_11)^(1/3))-17;
Uk11=13*Wk11*(uk11_-uk_);
Vk11=13*Wk11*(vk11_-vk_);

%% 12
S_refX12 = S_ref.*CMF_X.*TCS12;
S_refX12(isnan(S_refX12)) = 0;
XX_ref12 = trapz(wL*1E9,S_refX12);
 
S_refY12 = S_ref.*CMF_Y.*TCS12;
S_refY12 (isnan(S_refY12)) = 0;
YY_ref12 = trapz(wL*1E9,S_refY12);
 
S_refZ12 = S_ref.*CMF_Z.*TCS12;
S_refZ12(isnan(S_refZ12)) = 0;
ZZ_ref12 = trapz(wL*1E9, S_refZ12);
 
U_ref12=(2/3)*XX_ref12;
W_ref12=(1/2)*(-XX_ref12+3*YY_ref12+ZZ_ref12);
u_ref12=U_ref12/(U_ref12+YY_ref12+W_ref12);
v_ref12=YY_ref12/(U_ref12+YY_ref12+W_ref12);

TX12=T_tot.*CMF_X.*TCS12;
TX12(isnan(TX12)) = 0;
XX_12 = trapz(wL*1E9,TX12);
 
TY12 = T_tot.*CMF_Y.*TCS12;
TY12(isnan(TY12)) = 0;
YY_12 = trapz(wL*1E9,TY12);
 
TZ12 = T_tot.*CMF_Z.*TCS12;
TZ12(isnan(TZ12)) = 0;
ZZ_12 = trapz(wL*1E9,TZ12);
 
U_12=(2/3)*XX_12;
W_12=(1/2)*(-XX_12+3*YY_12+ZZ_12);
u_t12=U_12/(U_12+YY_12+W_12);
v_t12=YY_12/(U_12+YY_12+W_12);
 
c_k12=(1/v_t12)*(4-u_t12-10*v_t12);
d_k12=(1/v_t12)*(1.708*v_t12+0.404-1.481*u_t12);


uk12_=(10.872+0.404*(c_r/c_k)*c_k12-4*(d_r/d_k)*d_k12)/(16.518+1.481*(c_r/c_k)*c_k12-(d_r/d_k)*d_k12);
vk12_=(5.520)/(16.518+1.481*(c_r/c_k)*c_k12-(d_r/d_k)*d_k12);
Wr12=25*(((100/YY_ref)*YY_ref12)^(1/3))-17;
Ur12=13*Wr12*(u_ref12-u_ref);
Vr12=13*Wr12*(v_ref12-v_ref);
Wk12=25*(((100/YY_)*YY_12)^(1/3))-17;
Uk12=13*Wk12*(uk12_-uk_);
Vk12=13*Wk12*(vk12_-vk_);


%% 13
S_refX13 = S_ref.*CMF_X.*TCS13;
S_refX13(isnan(S_refX13)) = 0;
XX_ref13 = trapz(wL*1E9,S_refX13);
 
S_refY13 = S_ref.*CMF_Y.*TCS13;
S_refY13 (isnan(S_refY13)) = 0;
YY_ref13 = trapz(wL*1E9,S_refY13);
 
S_refZ13 = S_ref.*CMF_Z.*TCS13;
S_refZ13(isnan(S_refZ13)) = 0;
ZZ_ref13 = trapz(wL*1E9, S_refZ13);
 
U_ref13=(2/3)*XX_ref13;
W_ref13=(1/2)*(-XX_ref13+3*YY_ref13+ZZ_ref13);
u_ref13=U_ref13/(U_ref13+YY_ref13+W_ref13);
v_ref13=YY_ref13/(U_ref13+YY_ref13+W_ref13);

TX13=T_tot.*CMF_X.*TCS13;
TX13(isnan(TX13)) = 0;
XX_13 = trapz(wL*1E9,TX13);
 
TY13 = T_tot.*CMF_Y.*TCS13;
TY13(isnan(TY13)) = 0;
YY_13 = trapz(wL*1E9,TY13);
 
TZ13 = T_tot.*CMF_Z.*TCS13;
TZ13(isnan(TZ13)) = 0;
ZZ_13 = trapz(wL*1E9,TZ13);
 
U_13=(2/3)*XX_13;
W_13=(1/2)*(-XX_13+3*YY_13+ZZ_13);
u_t13=U_13/(U_13+YY_13+W_13);
v_t13=YY_13/(U_13+YY_13+W_13);
 
c_k13=(1/v_t13)*(4-u_t13-10*v_t13);
d_k13=(1/v_t13)*(1.708*v_t13+0.404-1.481*u_t13);


uk13_=(10.872+0.404*(c_r/c_k)*c_k13-4*(d_r/d_k)*d_k13)/(16.518+1.481*(c_r/c_k)*c_k13-(d_r/d_k)*d_k13);
vk13_=(5.520)/(16.518+1.481*(c_r/c_k)*c_k13-(d_r/d_k)*d_k13);
Wr13=25*(((100/YY_ref)*YY_ref13)^(1/3))-17;
Ur13=13*Wr13*(u_ref13-u_ref);
Vr13=13*Wr13*(v_ref13-v_ref);
Wk13=25*(((100/YY_)*YY_13)^(1/3))-17;
Uk13=13*Wk13*(uk13_-uk_);
Vk13=13*Wk13*(vk13_-vk_);

%% 14
S_refX14 = S_ref.*CMF_X.*TCS14;
S_refX14(isnan(S_refX14)) = 0;
XX_ref14 = trapz(wL*1E9,S_refX14);
 
S_refY14 = S_ref.*CMF_Y.*TCS14;
S_refY14 (isnan(S_refY14)) = 0;
YY_ref14 = trapz(wL*1E9,S_refY14);
 
S_refZ14 = S_ref.*CMF_Z.*TCS14;
S_refZ14(isnan(S_refZ14)) = 0;
ZZ_ref14 = trapz(wL*1E9, S_refZ14);
 
U_ref14=(2/3)*XX_ref14;
W_ref14=(1/2)*(-XX_ref14+3*YY_ref14+ZZ_ref14);
u_ref14=U_ref14/(U_ref14+YY_ref14+W_ref14);
v_ref14=YY_ref14/(U_ref14+YY_ref14+W_ref14);

TX14=T_tot.*CMF_X.*TCS14;
TX14(isnan(TX14)) = 0;
XX_14 = trapz(wL*1E9,TX14);
 
TY14 = T_tot.*CMF_Y.*TCS14;
TY14(isnan(TY14)) = 0;
YY_14 = trapz(wL*1E9,TY14);
 
TZ14 = T_tot.*CMF_Z.*TCS14;
TZ14(isnan(TZ14)) = 0;
ZZ_14 = trapz(wL*1E9,TZ14);
 
U_14=(2/3)*XX_14;
W_14=(1/2)*(-XX_14+3*YY_14+ZZ_14);
u_t14=U_14/(U_14+YY_14+W_14);
v_t14=YY_14/(U_14+YY_14+W_14);
 
c_k14=(1/v_t14)*(4-u_t14-10*v_t14);
d_k14=(1/v_t14)*(1.708*v_t14+0.404-1.481*u_t14);


uk14_=(10.872+0.404*(c_r/c_k)*c_k14-4*(d_r/d_k)*d_k14)/(16.518+1.481*(c_r/c_k)*c_k14-(d_r/d_k)*d_k14);
vk14_=(5.520)/(16.518+1.481*(c_r/c_k)*c_k14-(d_r/d_k)*d_k14);
Wr14=25*(((100/YY_ref)*YY_ref14)^(1/3))-17;
Ur14=13*Wr14*(u_ref14-u_ref);
Vr14=13*Wr14*(v_ref14-v_ref);
Wk14=25*(((100/YY_)*YY_14)^(1/3))-17;
Uk14=13*Wk14*(uk14_-uk_);
Vk14=13*Wk14*(vk14_-vk_);


%% 15
S_refX15 = S_ref.*CMF_X.*TCS15;
S_refX15(isnan(S_refX15)) = 0;
XX_ref15 = trapz(wL*1E9,S_refX15);
 
S_refY15 = S_ref.*CMF_Y.*TCS15;
S_refY15 (isnan(S_refY15)) = 0;
YY_ref15 = trapz(wL*1E9,S_refY15);
 
S_refZ15 = S_ref.*CMF_Z.*TCS15;
S_refZ15(isnan(S_refZ15)) = 0;
ZZ_ref15 = trapz(wL*1E9, S_refZ15);
 
U_ref15=(2/3)*XX_ref15;
W_ref15=(1/2)*(-XX_ref15+3*YY_ref15+ZZ_ref15);
u_ref15=U_ref15/(U_ref15+YY_ref15+W_ref15);
v_ref15=YY_ref15/(U_ref15+YY_ref15+W_ref15);

TX15=T_tot.*CMF_X.*TCS15;
TX15(isnan(TX15)) = 0;
XX_15 = trapz(wL*1E9,TX15);
 
TY15 = T_tot.*CMF_Y.*TCS15;
TY15(isnan(TY15)) = 0;
YY_15 = trapz(wL*1E9,TY15);
 
TZ15 = T_tot.*CMF_Z.*TCS15;
TZ15(isnan(TZ15)) = 0;
ZZ_15 = trapz(wL*1E9,TZ15);
 
U_15=(2/3)*XX_15;
W_15=(1/2)*(-XX_15+3*YY_15+ZZ_15);
u_t15=U_15/(U_15+YY_15+W_15);
v_t15=YY_15/(U_15+YY_15+W_15);
 
c_k15=(1/v_t15)*(4-u_t15-10*v_t15);
d_k15=(1/v_t15)*(1.708*v_t15+0.404-1.481*u_t15);


uk15_=(10.872+0.404*(c_r/c_k)*c_k15-4*(d_r/d_k)*d_k15)/(16.518+1.481*(c_r/c_k)*c_k15-(d_r/d_k)*d_k15);
vk15_=(5.520)/(16.518+1.481*(c_r/c_k)*c_k15-(d_r/d_k)*d_k15);
Wr15=25*(((100/YY_ref)*YY_ref15)^(1/3))-17;
Ur15=13*Wr15*(u_ref15-u_ref);
Vr15=13*Wr15*(v_ref15-v_ref);
Wk15=25*(((100/YY_)*YY_15)^(1/3))-17;
Uk15=13*Wk15*(uk15_-uk_);
Vk15=13*Wk15*(vk15_-vk_);

%% Calculation of color difference
Del_E1=sqrt(((Ur1-Uk1)^2)+((Vr1-Vk1)^2)+((Wr1-Wk1)^2));
Del_E2=sqrt(((Ur2-Uk2)^2)+((Vr2-Vk2)^2)+((Wr2-Wk2)^2));
Del_E3=sqrt(((Ur3-Uk3)^2)+((Vr3-Vk3)^2)+((Wr3-Wk3)^2));
Del_E4=sqrt(((Ur4-Uk4)^2)+((Vr4-Vk4)^2)+((Wr4-Wk4)^2));
Del_E5=sqrt(((Ur5-Uk5)^2)+((Vr5-Vk5)^2)+((Wr5-Wk5)^2));
Del_E6=sqrt(((Ur6-Uk6)^2)+((Vr6-Vk6)^2)+((Wr6-Wk6)^2));
Del_E7=sqrt(((Ur7-Uk7)^2)+((Vr7-Vk7)^2)+((Wr7-Wk7)^2));
Del_E8=sqrt(((Ur8-Uk8)^2)+((Vr8-Vk8)^2)+((Wr8-Wk8)^2));
Del_E9=sqrt(((Ur9-Uk9)^2)+((Vr9-Vk9)^2)+((Wr9-Wk9)^2));
Del_E10=sqrt(((Ur10-Uk10)^2)+((Vr10-Vk10)^2)+((Wr10-Wk10)^2));
Del_E11=sqrt(((Ur11-Uk11)^2)+((Vr11-Vk11)^2)+((Wr11-Wk11)^2));
Del_E12=sqrt(((Ur12-Uk12)^2)+((Vr12-Vk12)^2)+((Wr12-Wk12)^2));
Del_E13=sqrt(((Ur13-Uk13)^2)+((Vr13-Vk13)^2)+((Wr13-Wk13)^2));
Del_E14=sqrt(((Ur14-Uk14)^2)+((Vr14-Vk14)^2)+((Wr14-Wk14)^2));
Del_E15=sqrt(((Ur15-Uk15)^2)+((Vr15-Vk15)^2)+((Wr15-Wk15)^2));


%% Calculation of special rendering index Ri

R1=100-(4.6*Del_E1);
R2=100-(4.6*Del_E2);
R3=100-(4.6*Del_E3);
R4=100-(4.6*Del_E4);
R5=100-(4.6*Del_E5);
R6=100-(4.6*Del_E6);
R7=100-(4.6*Del_E7);
R8=100-(4.6*Del_E8);
R9=100-(4.6*Del_E9);
R10=100-(4.6*Del_E10);
R11=100-(4.6*Del_E11);
R12=100-(4.6*Del_E12);
R13=100-(4.6*Del_E13);
R14=100-(4.6*Del_E14);
R15=100-(4.6*Del_E15);

 CRI_gen = (R1+R2+R3+R4+R5+R6+R7+R8)/8;
 CRI_ext = (R1+R2+R3+R4+R5+R6+R7+R8+R9+R10+R11+R12+R13+R14)/14;


%% Jph


 active_layer = load([path 'PTB7_PCBM' '.txt']);
 d_al = 100E-9; % nm
 h = 4.13566766225e-15; % Planck's constant, h. [eVs]
 c = 2.9979e8; % Speed of light in vacuum. [m/s]

 k_al = interp1(active_layer(:,1), active_layer(:,3), wL*1E6, 'linear');
 k_al(isnan(k_al)) = 0;

 alpha_al = (4*pi)./wL.*k_al;

 
%  Jph_integrant = T_tot.*S_AM15G.*(1-exp(-1.*alpha_al*d_al));
   Jph_integrant = A_tot.*S_AM15G;   % Jph (A_tot)


Jph_integrant(isnan(Jph_integrant)) = 0;
Jph = (1/(h*c))*trapz(wL*1E9,Jph_integrant)*100*1E-9; % mA/cm^2

%%
 wL = wL*1e9;
 maxT = max(T_tot);
 num = [wL,R_tot,T_tot,A_tot];
 [~,z] = max(num(:,3));
 maxwL = num(z,1);
 
 
  result(m,1)=m-1;
  result(m,2)=thickness(1);
  result(m,3)=thickness(2);
  result(m,4)=thickness(3);
  result(m,5)=thickness(4);
  result(m,6)=thickness(5);
  result(m,7)=thickness(6);
  result(m,8)=thickness(7);
  result(m,9)=thickness(8);
  result(m,10)=AVT;
  result(m,11)=x_cr;
  result(m,12)=y_cr;
  result(m,13)=CRI_ext;
  result(m,14)=CCT;
  result(m,15)=DC;
  result(m,16)=Jph;
  result(m,17)=maxT;
  result(m,18)=maxwL;


  m=m+1;
        end

    end

% end

 %        end
 % 
 %    end
 % 
 % end
 %% Maximum Value

 [~,q] = max(result(:,10));
 maxAVT = result(q,:);

 [~,w] = max(result(:,16));
 maxJph = result(w,:);

 [~,x] = max(result(:,13));
 maxCRI_ext = result(x,:);


 %% Print
  
   headery = {'Design'};
   column_number = size(layers,2);
   structures = cell(column_number,1);
   structures_t = cell(column_number,2);
   structures = layers;
   structures_t = thickness;
   sheet2 = cell(column_number,2);
   bb(1,:) = structures;
   headerx = horzcat(headery, bb);
   
 
   header1 = {'AVT','x  ','y  ','CRI_ext','CCT     ','Chromacity Distance','Jph (mA/cm^2)','MaxT','MaxwL (nm)'};
   data = num2cell([result]);
   header = horzcat(headerx, header1);

   data(1,:) = header;
  
   writecell(data,filename,'Sheet',1);


  %% Print - Sheet2

   
   header2 = {'STR-1 (AVT-Max)','STR-2 (Jph-Max)','STR-3 (CRI-Max)'};
   
   
   sheet2 = cell(4,12);
   
   sheet2(1,2:19)=header;
   sheet2(2:4,1)=header2;
   sheet2(2,2:19)=num2cell([maxAVT]);
   sheet2(3,2:19)=num2cell([maxJph]);
   sheet2(4,2:19)=num2cell([maxCRI_ext]);
   
   
   
   
   writecell(sheet2,filename,'Sheet',2);


