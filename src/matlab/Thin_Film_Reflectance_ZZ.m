function [R_s,A_2] = Thin_Film_Reflectance_ZZ(wavelength, thicknesses,layers,angle)
% INPUT: Wavelength (in m), Layer thicknesses (in m), refractive_indices (layer/column), angle (degrees)
% OUTPUT: Reflectance (R), p-pol R, s-pol R
% COMMENTS: This function does not load and interpolate refractive indices 
% of the specific materials. 

n = length(layers(1,:));
N = layers;
wL = wavelength;
d = thicknesses;
theta = angle*pi()/180;
c = 2.9979e8; % speed of ligth in vacuo m/s

eps = N.^2;

% o = zeros(length(n),1);
% p_p = zeros(length(n),1);
% p_s = zeros(length(n),1);
% phi = zeros(length(n),1);
% R_p = zeros(length(wL),1);
R_s = zeros(length(wL),1);
A_2 = zeros(length(wL),1);

% s or TE polarized 
for w=1:length(wL)
%layer 1
k0 = 2*pi/wL(w);
kx = k0*sin(theta);
kz = sqrt(eps(w,:).*k0.^2 - kx^2);

P_1 = [1 0;0 1];
D_1 = [1 1;kz(1) -kz(1)];

P_2 = [exp(-i*kz(2)*d(2)) 0 ; 0 exp(i*kz(2)*d(2))];
D_2 = [1 1 ; kz(2) -kz(2)];

D_3 = [1 1 ; kz(3) -kz(3)];

M = P_1*D_1^(-1)*D_2*P_2*D_2^(-1)*D_3;

r_s = M(2,1)./M(1,1);
R_s(w) = (abs(r_s)).^2;

T_2 = P_2*D_2^(-1)*D_3;

S_1f = cos(theta);
S_1b =  (abs(M(2,1)./M(1,1))^2)* cos(theta);
theta_2 = asin(N(w,1)*sin(theta)/N(w,2));
S_2f = real(N(w,2))*abs(T_2(1,1)./M(1,1))^2 * cos(theta_2);
S_2b = real(N(w,2))*abs(T_2(2,1)./M(1,1))^2 * cos(theta_2);

A_2(w) = ( S_1f - S_1b - (S_2f - S_2b) )/S_1f;

end




end

