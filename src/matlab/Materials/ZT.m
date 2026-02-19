function [refractive_index,rel_permittivity] = ZT(wavelength_range, PLOT_ON)
% INPUT: Wavelength = meters [short L limit, long L limit]
% OUTPUT: [refractive_index , rel_permittivity]
% Range of validity: 0.2-2 microns
% Paded w/ constants outside this range

if nargin < 3 || isempty(PLOT_ON),          PLOT_ON = 0;      end

path = [cd '/Materials/Properties/'];

wL = wavelength_range*1e6; 
wL_limit_short = 0.2;
wL_limit_long  = 2;
 
layer = load([path 'ZT' '.txt']);


n_j = interp1(layer(:,1), layer(:,2), wL, 'linear');
k_j = interp1(layer(:,1), layer(:,3), wL, 'linear');


N = n_j + 1i*k_j;   

refractive_index = N;
eps = N.^2;
rel_permittivity = eps;

if PLOT_ON
    
figure,
loglog(layer(:,1),[layer(:,2) layer(:,3)],'x'); 
hold all
loglog(wL,[real(N)' imag(N)']); 
xlim([min(wL) max(wL)])

figure,
plot(wL,[real(eps)' imag(eps)']); 
xlim([min(wL) max(wL)])
end
 
end

