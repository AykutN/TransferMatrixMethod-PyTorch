function [Rs,Rp,Ts,Tp,angle_out] = TMM_Ref_Trans_Thick(wavelength, thicknesses,layers,angle)
% INPUT: Wavelength (in m), Layer thicknesses (in m), refractive_indices (layer/column), angle (degrees)
% OUTPUT: Reflectance (R), p-pol R, s-pol R
% COMMENTS: This function does not load and interpolate refractive indices 
% of the specific materials. 

n = length(layers(1,:));
N = layers;
wL = wavelength;
d = thicknesses;
c = 2.9979e8; % speed of ligth in vacuo m/s
o = zeros(length(n),1);
p_p = zeros(length(n),1);
p_s = zeros(length(n),1);
phi = zeros(length(n),1);
R12p = zeros(length(wL),1);
R12s = zeros(length(wL),1);
T12p = zeros(length(wL),1);
T12s = zeros(length(wL),1);

for w=1:length(wL)
%layer 1
M_p = 1;
M_s = 1;
o(1) = angle*pi()/180;
p_p(1) = cos(o(1)) ./ (N(w,1)./c);
p_s(1) = cos(o(1)) ./ (-c./N(w,1));

if n <= 2
j = 2;    
o(j) = asin(N(w,1)./N(w,j).*sin(o(1)));
L1 =  1;
L2 =  2;

r12p = (N(w,L2)*cos(o(L1))- N(w,L1)*cos(o(L2))) / (N(w,L1)*cos(o(L2))+ N(w,L2)*cos(o(L1)));
r12s = (N(w,L1)*cos(o(L1))- N(w,L2)*cos(o(L2))) / (N(w,L1)*cos(o(L1))+ N(w,L2)*cos(o(L2)));

R12p(w) = r12p.*conj(r12p);
R12s(w) = r12s.*conj(r12s);

t12p = 2*N(w,L1).*cos(o(L1))./(N(w,L2).*cos(o(L1)) + N(w,L1).*cos(o(L2)));
t12s = 2*N(w,L1).*cos(o(L1))./(N(w,L1).*cos(o(L1)) + N(w,L2).*cos(o(L2)));

T12s(w) = real(N(w,L2) .* cos(o(L2)))./real(N(w,L1).*cos(o(L1))).*t12s.*conj(t12s);
T12p(w) = real(N(w,L2) .* cos(o(L2)))./real(N(w,L1).*cos(o(L1))).*t12p.*conj(t12p);
else
   
%layer j
for j=2:(n-1)
o(j) = asin(N(w,1)./N(w,j).*sin(o(1)));
p_p(j) = cos(o(j)) ./ (N(w,j)./c);
p_s(j) = cos(o(j)) ./ (-c./N(w,j));
phi(j) = 2*pi*d(j).*N(w,j).*cos(o(j))./wL(w);


M_pj = [cos(phi(j)) -1i.*p_p(j).*sin(phi(j)); -1i./p_p(j).*sin(phi(j)) cos(phi(j))];
M_sj = [cos(phi(j)) -1i.*p_s(j).*sin(phi(j)); -1i./p_s(j).*sin(phi(j)) cos(phi(j))];

M_p = M_p*M_pj;
M_s = M_s*M_sj;

end

%layer n
o(n) = asin(N(w,1)./N(w,n).*sin(o(1)));
p_p(n) = cos(o(n)) ./ (N(w,n)./c);
p_s(n) = cos(o(n)) ./ (-c./N(w,n));

% p or TM polarization
r_p = ((M_p(1,1) + 1./p_p(n).*M_p(1,2)) - (M_p(2,1) + 1./p_p(n).*M_p(2,2)).*p_p(1))./...
    ((M_p(1,1) + 1./p_p(n).*M_p(1,2))+(M_p(2,1) + 1./p_p(n).*M_p(2,2)).*p_p(1));
R12p(w) = r_p*conj(r_p);

c_p = cos(o(1))/cos(o(n));

t_p = 2*c_p /((M_p(1,1) + 1./p_p(n).*M_p(1,2))+(M_p(2,1) + 1./p_p(n).*M_p(2,2)).*p_p(1));

T12p(w) = real((N(w,n)) * cos(o(n))) / real((N(w,1)) * cos(o(1))) * (t_p*conj(t_p));

% s or TE polarization
r_s = ((M_s(1,1) + 1./p_s(n).*M_s(1,2)) - (M_s(2,1) + 1./p_s(n).*M_s(2,2)).*p_s(1))./...
    ((M_s(1,1) + 1./p_s(n).*M_s(1,2))+(M_s(2,1) + 1./p_s(n).*M_s(2,2)).*p_s(1));
R12s(w) = r_s*conj(r_s);

t_s = 2 / ((M_s(1,1) + 1./p_s(n).*M_s(1,2))+(M_s(2,1) + 1./p_s(n).*M_s(2,2)).*p_s(1))*N(w,1)/N(w,n)*cos(o(1))/cos(o(n));
% t_s = 2 / ((M_s(1,1) + 1./p_s(n).*M_s(1,2))+(M_s(2,1) + 1./p_s(n).*M_s(2,2)).*p_s(1));
T12s(w) = real(N(w,n) * cos(o(n))) / real(N(w,1) * cos(o(1))) * (t_s*conj(t_s));

end

end

Rs = real(R12s);
Rp = real(R12p);
Ts = real(T12s);
Tp = real(T12p);
angle_out = o(end)*180/pi;
end