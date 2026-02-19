function [Reflectance,R_p,R_s] = Thin_Film_Reflectance(wavelength, thicknesses,layers,angle)
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
R_p = zeros(length(wL),1);
R_s = zeros(length(wL),1);

for w=1:length(wL)


%layer 1
M_p = 1;
M_s = 1;
o(1) = angle*pi()/180;
p_p(1) = cos(o(1)) ./ (N(w,1)./c);
p_s(1) = cos(o(1)) ./ (-c./N(w,1));

if n <= 2
j = 2;
o(j) = asin(N(w,j-1)./N(w,j).*sin(o(j-1)));
L1 =  1;
L2 =  2;

rp = (N(w,L1)*cos(o(L2))- N(w,L2)*cos(o(L1)))/(N(w,L1)*cos(o(L2))+ N(w,L2)*cos(o(L1)));
rs = (N(w,L1)*cos(o(L1))- N(w,L2)*cos(o(L2)))/(N(w,L1)*cos(o(L1))+ N(w,L2)*cos(o(L2)));

R_p(w) = (abs(rp)).^2;
R_s(w) = (abs(rs)).^2;

else
    
%layer j
for j=2:(n-1)
o(j) = asin(N(w,j-1)./N(w,j).*sin(o(j-1)));
p_p(j) = cos(o(j)) ./ (N(w,j)./c);
p_s(j) = cos(o(j)) ./ (-c./N(w,j));
phi(j) = 2*pi*d(j).*N(w,j).*cos(o(j))./wL(w);


M_pj = [cos(phi(j)) -1i.*p_p(j).*sin(phi(j)); -1i./p_p(j).*sin(phi(j)) cos(phi(j))];
M_sj = [cos(phi(j)) -1i.*p_s(j).*sin(phi(j)); -1i./p_s(j).*sin(phi(j)) cos(phi(j))];

M_p = M_p*M_pj;
M_s = M_s*M_sj;

end

%layer n
o(n) = asin(N(w,n-1)./N(w,n).*sin(o(n-1)));
p_p(n) = cos(o(n)) ./ (N(w,n)./c);
p_s(n) = cos(o(n)) ./ (-c./N(w,n));

% p or TM polarization
r_p = ((M_p(1,1) + 1./p_p(n).*M_p(1,2)) - (M_p(2,1) + 1./p_p(n).*M_p(2,2)).*p_p(1))./...
    ((M_p(1,1) + 1./p_p(n).*M_p(1,2))+(M_p(2,1) + 1./p_p(n).*M_p(2,2)).*p_p(1));
R_p(w) = (abs(r_p)).^2;

% s or TE polarization
r_s = ((M_s(1,1) + 1./p_s(n).*M_s(1,2)) - (M_s(2,1) + 1./p_s(n).*M_s(2,2)).*p_s(1))./...
    ((M_s(1,1) + 1./p_s(n).*M_s(1,2))+(M_s(2,1) + 1./p_s(n).*M_s(2,2)).*p_s(1));
R_s(w) = (abs(r_s)).^2;


end

end


Reflectance =  0.5.*(R_p + R_s);

end

